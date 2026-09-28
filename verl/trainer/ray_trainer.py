# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.dataset import collate_fn
from ..utils.logger import Tracker
from ..utils.multimodal_contract import validate_multi_modal_data_contract
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


if TYPE_CHECKING:
    from ..workers.fsdp_workers import FSDPWorker


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

        # === Review Queue (V3: pre-replacement) ===
        self._current_review_uids: set = set()
        self._current_review_pid_by_uid: dict = {}
        if config.algorithm.review_queue_enabled:
            self.review_queue = deque(maxlen=config.algorithm.review_queue_max_size)
            self.review_queue_set: set = set()
            self.train_dataset = train_dataloader.dataset

            self.review_log: list[dict] = []
            self.review_log_path = os.path.join(
                config.trainer.save_checkpoint_path, "review_log.json"
            )

            self.current_review_freq: int = config.algorithm.review_freq
            if config.algorithm.adaptive_review_freq:
                self.regression_rate_window: deque = deque(
                    maxlen=config.algorithm.adaptive_freq_window
                )

            if config.algorithm.prioritized_replay:
                self.regression_count: dict[int, int] = {}

            # Monitor cohort
            self.monitor_pids: list[int] = []
            self.monitor_pid_set: set = set()
            self.monitor_collecting = config.algorithm.monitor_cohort_enabled
            self.monitor_collect_until_step = 2
            self.monitor_log: list[dict] = []
            self.monitor_path = os.path.join(
                config.trainer.save_checkpoint_path, "monitor_cohort_tracking.json"
            )

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

        # Save review queue state
        if self.config.algorithm.review_queue_enabled:
            review_queue_path = os.path.join(folder_path, "review_queue.json")
            with open(review_queue_path, "w") as f:
                json.dump(list(self.review_queue), f)

            if self.config.algorithm.prioritized_replay and hasattr(self, "regression_count"):
                regression_count_path = os.path.join(folder_path, "review_regression_count.json")
                with open(regression_count_path, "w") as f:
                    json.dump(self.regression_count, f)

            if self.config.algorithm.adaptive_review_freq and hasattr(self, "regression_rate_window"):
                adaptive_freq_path = os.path.join(folder_path, "review_adaptive_freq.json")
                with open(adaptive_freq_path, "w") as f:
                    json.dump({
                        "current_review_freq": self.current_review_freq,
                        "regression_rate_window": list(self.regression_rate_window),
                    }, f)

            # Persist logs
            try:
                os.makedirs(os.path.dirname(self.review_log_path), exist_ok=True)
                with open(self.review_log_path, "w") as f:
                    json.dump(self.review_log, f, indent=2, ensure_ascii=False)
                with open(self.monitor_path, "w") as f:
                    json.dump(self.monitor_log, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ReviewQueue] Warning: failed to save logs: {e}")

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

        # Restore review queue state
        if self.config.algorithm.review_queue_enabled:
            review_queue_path = os.path.join(load_checkpoint_path, "review_queue.json")
            if os.path.exists(review_queue_path):
                with open(review_queue_path, "r") as f:
                    saved_ids = json.load(f)
                self.review_queue.clear()
                for pid in saved_ids:
                    self.review_queue.append(pid)
                self.review_queue_set = set(self.review_queue)
                print(f"[ReviewQueue] Restored {len(self.review_queue)} items from checkpoint.")

            if self.config.algorithm.prioritized_replay:
                regression_count_path = os.path.join(load_checkpoint_path, "review_regression_count.json")
                if os.path.exists(regression_count_path):
                    with open(regression_count_path, "r") as f:
                        loaded = json.load(f)
                    self.regression_count = {int(k): v for k, v in loaded.items()}
                    print(f"[ReviewQueue] Restored regression_count for {len(self.regression_count)} items.")

            if self.config.algorithm.adaptive_review_freq:
                adaptive_freq_path = os.path.join(load_checkpoint_path, "review_adaptive_freq.json")
                if os.path.exists(adaptive_freq_path):
                    with open(adaptive_freq_path, "r") as f:
                        state = json.load(f)
                    self.current_review_freq = state.get("current_review_freq", self.config.algorithm.review_freq)
                    window_data = state.get("regression_rate_window", [])
                    self.regression_rate_window = deque(
                        window_data, maxlen=self.config.algorithm.adaptive_freq_window
                    )
                    print(f"[ReviewQueue] Restored adaptive freq={self.current_review_freq}.")

    def _assert_multimodal_contract(self, data: DataProto, stage: str) -> None:
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        problem_ids = data.non_tensor_batch.get("problem_id", None)
        uids = data.non_tensor_batch.get("uid", None)
        for idx, multi_modal_data in enumerate(data.non_tensor_batch["multi_modal_data"]):
            try:
                validate_multi_modal_data_contract(multi_modal_data)
            except Exception as exc:
                problem_id = None if problem_ids is None else problem_ids[idx]
                uid = None if uids is None else uids[idx]
                raise ValueError(
                    f"{stage}: invalid multi_modal_data at index={idx}, uid={uid}, problem_id={problem_id}: {exc}"
                ) from exc

    def _maybe_log_val_generations(
        self,
        inputs: list[str],
        outputs: list[str],
        labels: list[str],
        scores: list[float],
        problem_ids: list[Any],
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, label, score, problem_id) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores, problem_ids))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores, sample_problem_ids = [], [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["image_min_pixels"] = self.config.data.image_min_pixels
            test_gen_batch.meta_info["image_max_pixels"] = self.config.data.image_max_pixels
            test_gen_batch.meta_info["video_min_pixels"] = self.config.data.val_video_min_pixels
            test_gen_batch.meta_info["video_max_pixels"] = self.config.data.val_video_max_pixels
            test_gen_batch.meta_info["video_total_pixels"] = self.config.data.val_video_total_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.val_video_fps
            test_gen_batch.meta_info["video_max_frames"] = self.config.data.val_video_max_frames

            self._assert_multimodal_contract(test_gen_batch, stage="validate")
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # Only pass fields needed by reward, excluding large multi_modal_data to reduce serialization
            val_reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(val_reward_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)
            if "problem_id" in test_batch.non_tensor_batch:
                sample_problem_ids.extend(test_batch.non_tensor_batch["problem_id"].tolist())
            else:
                sample_problem_ids.extend([None] * len(scores))

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(
            sample_inputs, sample_outputs, sample_labels, sample_scores, sample_problem_ids
        )
        if self.config.trainer.val_generations_to_log > 0 and sample_inputs:
            print("Sample problem_id:", sample_problem_ids[0])
            print("Sample prompt (with template):", sample_inputs[0])
            print("Sample response:", sample_outputs[0])
            print("Sample ground_truth:", sample_labels[0])
            print("Sample reward:", sample_scores[0])
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _mix_offline_trajectories_in_gen_output(
        self,
        gen_batch_output: DataProto,
        batch_has_offline: Optional[np.ndarray],
        batch_offline_output: Optional[np.ndarray],
        replace_offline_mask: np.ndarray,
        n: int,
    ) -> int:
        """Replace the last rollout in each selected group with offline text, then rebuild sequence tensors."""
        if batch_has_offline is None or batch_offline_output is None or len(replace_offline_mask) == 0:
            return 0

        responses = gen_batch_output.batch["responses"].clone()
        response_length = responses.size(1)
        device = responses.device
        pad_token_id = self.tokenizer.pad_token_id
        replaced = 0

        for i, use_offline in enumerate(replace_offline_mask):
            if not use_offline:
                continue
            if not bool(batch_has_offline[i]):
                continue

            offline_output = batch_offline_output[i]
            if not isinstance(offline_output, str) or len(offline_output.strip()) == 0:
                continue

            offline_token_ids = self.tokenizer.encode(offline_output, add_special_tokens=False)
            offline_tokens = VF.pad_2d_list_to_length(
                [offline_token_ids],
                pad_token_id,
                max_length=response_length,
            ).to(device)

            replace_idx = i * n + (n - 1)
            responses[replace_idx] = offline_tokens.squeeze(0)
            replaced += 1

        if replaced == 0:
            return 0

        prompts = gen_batch_output.batch["prompts"]
        prompt_length = prompts.size(-1)
        attention_mask = gen_batch_output.batch["attention_mask"]
        position_ids = gen_batch_output.batch["position_ids"]
        eos_token_id = gen_batch_output.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)

        prompt_attention_mask = attention_mask[..., :prompt_length]
        prompt_position_ids = position_ids[..., :prompt_length]

        response_mask = VF.get_response_mask(responses, eos_token_id=eos_token_id, dtype=prompt_attention_mask.dtype)
        sequence_ids = torch.cat([prompts, responses], dim=-1)

        batch_size = responses.size(0)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if prompt_position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, prompt_position_ids.size(1), -1
            )

        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        full_attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        gen_batch_output.batch["responses"] = responses
        gen_batch_output.batch["input_ids"] = sequence_ids
        gen_batch_output.batch["response_mask"] = response_mask
        gen_batch_output.batch["position_ids"] = full_position_ids
        gen_batch_output.batch["attention_mask"] = full_attention_mask
        return replaced

    # ==================== Review Queue (V3) ====================

    def _is_review_step(self) -> bool:
        """Check if the current step should perform review replacement."""
        if not self.config.algorithm.review_queue_enabled:
            return False
        if self.global_step < self.config.algorithm.review_start_step:
            return False
        freq = getattr(self, "current_review_freq", self.config.algorithm.review_freq)
        return self.global_step % freq == 0

    def _prepare_review_replacement(self, new_batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Replace the tail of new_batch with review queue samples (pre-rollout).

        This is the core V3 change: review samples enter the main batch BEFORE
        rollout, so they share the same generate_sequences + compute_reward pipeline.
        No extra rollout cost.

        Returns the modified batch. Sets self._current_review_uids and
        self._current_review_pid_by_uid for post-reward processing.
        """
        self._current_review_uids = set()
        self._current_review_pid_by_uid = {}

        if not self._is_review_step() or len(self.review_queue) == 0:
            return new_batch

        cfg = self.config.algorithm
        batch_size = len(new_batch.batch)
        replace_count = min(
            int(batch_size * cfg.review_replace_ratio),
            len(self.review_queue),
            batch_size,
        )

        if replace_count == 0:
            return new_batch

        # Sample from queue (prioritized or FIFO)
        if cfg.prioritized_replay and len(self.review_queue) > 0:
            queue_list = list(self.review_queue)
            weights = np.array([
                cfg.priority_base_weight + self.regression_count.get(pid, 0) * cfg.priority_count_weight
                for pid in queue_list
            ], dtype=np.float64)
            weights /= weights.sum()
            actual_pop = min(replace_count, len(queue_list))
            chosen_indices = np.random.choice(len(queue_list), size=actual_pop, replace=False, p=weights)
            popped_ids = [queue_list[i] for i in chosen_indices]
            popped_set = set(popped_ids)
            self.review_queue = deque(
                (pid for pid in self.review_queue if pid not in popped_set),
                maxlen=cfg.review_queue_max_size,
            )
            for pid in popped_ids:
                self.review_queue_set.discard(pid)
        else:
            popped_ids = []
            actual_pop = min(replace_count, len(self.review_queue))
            for _ in range(actual_pop):
                pid = self.review_queue.popleft()
                self.review_queue_set.discard(pid)
                popped_ids.append(pid)

        if len(popped_ids) == 0:
            return new_batch

        # Fetch items from dataset
        items = self.train_dataset.get_items_by_problem_ids(popped_ids)
        if len(items) == 0:
            # Re-enqueue if lookup failed
            for pid in popped_ids:
                if pid not in self.review_queue_set:
                    self.review_queue.append(pid)
                    self.review_queue_set.add(pid)
            return new_batch

        # Build review batch
        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_total_pixels": self.config.data.video_total_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        review_batch_dict = collate_fn(items)
        review_batch = DataProto.from_single_dict(review_batch_dict, meta_info=meta_info)
        review_uids = np.array(
            [str(uuid.uuid4()) for _ in range(len(review_batch.batch))], dtype=object
        )
        review_batch.non_tensor_batch["uid"] = review_uids

        # Map review UIDs to problem_ids for post-reward processing
        review_pids = review_batch.non_tensor_batch.get("problem_id")
        for i, uid in enumerate(review_uids):
            self._current_review_uids.add(uid)
            if review_pids is not None:
                pid = review_pids[i]
                if hasattr(pid, "item"):
                    pid = pid.item()
                self._current_review_pid_by_uid[uid] = pid

        # Replace the tail of new_batch with review samples
        actual_replace = len(review_batch.batch)
        keep_count = batch_size - actual_replace
        if keep_count > 0:
            kept_batch = new_batch[:keep_count]
            new_batch = DataProto.concat([kept_batch, review_batch])
        else:
            new_batch = review_batch

        metrics["review/replaced_count"] = actual_replace
        print(f"[ReviewQueue] Replaced {actual_replace} prompts in batch "
              f"(queue_size={len(self.review_queue)})")
        return new_batch

    def _process_review_results(
        self, reward_metrics: dict, batch: DataProto, metrics: dict[str, Any]
    ) -> None:
        """After reward, process review results and collect new mastered samples.

        For review samples:
          - All n rollouts correct -> graduate (stays out of queue)
          - Any rollout wrong -> re-enqueue (with incremented regression_count)

        For non-review samples:
          - All n rollouts correct -> enqueue as new mastered (with sample_rate)
        """
        accuracy_scores = reward_metrics.get("accuracy")
        if accuracy_scores is None:
            return

        uids = batch.non_tensor_batch["uid"]
        problem_ids = batch.non_tensor_batch.get("problem_id")

        # Group accuracy by UID
        uid2acc = defaultdict(list)
        uid_to_first_idx = {}
        for idx, (uid, acc) in enumerate(zip(uids, accuracy_scores)):
            uid2acc[uid].append(acc)
            if uid not in uid_to_first_idx:
                uid_to_first_idx[uid] = idx

        # --- Process review samples ---
        reviewed_count = 0
        graduated_count = 0
        regressed_count = 0
        regressed_pids = []

        # Preserve FIFO order when several reviewed prompts regress together.
        for uid in self._current_review_pid_by_uid:
            if uid not in uid2acc:
                # Review sample was filtered out by online_filtering (all correct)
                # Treat as graduated
                graduated_count += 1
                reviewed_count += 1
                pid = self._current_review_pid_by_uid.get(uid)
                if pid is not None and self.config.algorithm.prioritized_replay:
                    self.regression_count.pop(pid, None)
                continue

            reviewed_count += 1
            acc_list = uid2acc[uid]
            pid = self._current_review_pid_by_uid.get(uid)

            if all(a >= 1.0 for a in acc_list):
                # Still mastered -> graduate
                graduated_count += 1
                if pid is not None and self.config.algorithm.prioritized_replay:
                    self.regression_count.pop(pid, None)
            else:
                # Regressed -> re-enqueue
                regressed_count += 1
                if pid is not None:
                    regressed_pids.append(pid)
                    if self.config.algorithm.prioritized_replay:
                        self.regression_count[pid] = self.regression_count.get(pid, 0) + 1
                    if pid not in self.review_queue_set:
                        if len(self.review_queue) == self.review_queue.maxlen:
                            evicted = self.review_queue[0]
                            self.review_queue_set.discard(evicted)
                            if self.config.algorithm.prioritized_replay:
                                self.regression_count.pop(evicted, None)
                        self.review_queue.append(pid)
                        self.review_queue_set.add(pid)

        retention_rate = (
            1.0 - (regressed_count / reviewed_count) if reviewed_count > 0 else 1.0
        )

        # --- Collect new mastered samples (non-review) ---
        cfg = self.config.algorithm
        sample_rate = cfg.review_queue_sample_rate
        newly_added = 0
        for uid, acc_list in uid2acc.items():
            if uid in self._current_review_uids:
                continue
            if not all(a >= 1.0 for a in acc_list):
                continue
            first_idx = uid_to_first_idx.get(uid)
            if first_idx is None or problem_ids is None:
                continue
            pid = problem_ids[first_idx]
            if hasattr(pid, "item"):
                pid = pid.item()
            if pid in self.monitor_pid_set:
                continue
            if pid in self.review_queue_set:
                continue
            if sample_rate < 1.0 and np.random.random() >= sample_rate:
                continue
            if len(self.review_queue) == self.review_queue.maxlen:
                evicted = self.review_queue[0]
                self.review_queue_set.discard(evicted)
                if cfg.prioritized_replay:
                    self.regression_count.pop(evicted, None)
            self.review_queue.append(pid)
            self.review_queue_set.add(pid)
            if cfg.prioritized_replay and pid not in self.regression_count:
                self.regression_count[pid] = 0
            newly_added += 1

        # --- Monitor cohort collection ---
        if self.monitor_collecting:
            self._collect_monitor_samples(reward_metrics, batch)

        # --- Update adaptive frequency ---
        if reviewed_count > 0 and cfg.adaptive_review_freq:
            self._update_adaptive_freq(
                regressed_count / reviewed_count if reviewed_count > 0 else 0.0,
                metrics,
            )

        # --- Log ---
        metrics["review/queue_size"] = len(self.review_queue)
        metrics["review/new_collected"] = newly_added
        if reviewed_count > 0:
            metrics["review/reviewed_count"] = reviewed_count
            metrics["review/graduated_count"] = graduated_count
            metrics["review/regressed_count"] = regressed_count
            metrics["review/retention_rate"] = retention_rate

        if cfg.prioritized_replay and self.regression_count:
            metrics["review/avg_regression_count"] = (
                sum(self.regression_count.values()) / len(self.regression_count)
            )

        record = {
            "global_step": self.global_step,
            "queue_size": len(self.review_queue),
            "reviewed": reviewed_count,
            "graduated": graduated_count,
            "regressed": regressed_count,
            "retention_rate": round(retention_rate, 4) if reviewed_count > 0 else None,
            "new_collected": newly_added,
            "regressed_problem_ids": regressed_pids,
        }
        self.review_log.append(record)

        if reviewed_count > 0:
            print(
                f"[ReviewQueue] reviewed={reviewed_count}, graduated={graduated_count}, "
                f"regressed={regressed_count}, retention={retention_rate:.2%}, "
                f"new_collected={newly_added}, queue_size={len(self.review_queue)}"
            )
        else:
            print(f"[ReviewQueue] new_collected={newly_added}, queue_size={len(self.review_queue)}")

        # Reset per-step state
        self._current_review_uids = set()
        self._current_review_pid_by_uid = {}

    def _update_adaptive_freq(self, regression_rate: float, metrics: dict[str, Any]) -> None:
        """Update current_review_freq based on sliding window of regression rates."""
        cfg = self.config.algorithm
        self.regression_rate_window.append(regression_rate)
        avg_rate = sum(self.regression_rate_window) / len(self.regression_rate_window)

        if avg_rate > cfg.adaptive_freq_high_threshold:
            self.current_review_freq = cfg.review_freq_min
        elif avg_rate < cfg.adaptive_freq_low_threshold:
            self.current_review_freq = cfg.review_freq_max
        else:
            t = (avg_rate - cfg.adaptive_freq_low_threshold) / (
                cfg.adaptive_freq_high_threshold - cfg.adaptive_freq_low_threshold
            )
            self.current_review_freq = round(
                cfg.review_freq_max - t * (cfg.review_freq_max - cfg.review_freq_min)
            )

        metrics["review/current_freq"] = self.current_review_freq
        metrics["review/avg_regression_rate"] = avg_rate

    def _collect_monitor_samples(self, reward_metrics: dict, batch: DataProto) -> None:
        """Collect all-correct samples from the first 2 steps into a fixed monitor cohort."""
        if not self.monitor_collecting:
            return
        if self.global_step > self.monitor_collect_until_step:
            self.monitor_collecting = False
            print(f"[Monitor] collection complete: cohort_size={len(self.monitor_pids)}")
            return

        accuracy_scores = reward_metrics.get("accuracy")
        if accuracy_scores is None:
            return

        uids = batch.non_tensor_batch["uid"]
        problem_ids = batch.non_tensor_batch.get("problem_id")
        if problem_ids is None:
            return

        uid2acc = defaultdict(list)
        uid_to_first_idx = {}
        for idx, (uid, acc) in enumerate(zip(uids, accuracy_scores)):
            uid2acc[uid].append(acc)
            if uid not in uid_to_first_idx:
                uid_to_first_idx[uid] = idx

        newly_added = 0
        for uid, acc_list in uid2acc.items():
            if all(a >= 1.0 for a in acc_list):
                first_idx = uid_to_first_idx.get(uid)
                if first_idx is None:
                    continue
                pid = problem_ids[first_idx]
                if hasattr(pid, "item"):
                    pid = pid.item()
                if pid not in self.monitor_pid_set:
                    self.monitor_pids.append(pid)
                    self.monitor_pid_set.add(pid)
                    newly_added += 1

        if newly_added > 0:
            print(
                f"[Monitor] collected {newly_added} samples at step {self.global_step}, "
                f"total cohort_size={len(self.monitor_pids)}"
            )

    def _evaluate_monitor_cohort(self, metrics: dict[str, Any]) -> None:
        """Evaluate the fixed monitor cohort to track regression over training."""
        n = self.config.worker.rollout.n
        items = self.train_dataset.get_items_by_problem_ids(self.monitor_pids)
        if len(items) == 0:
            return

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_total_pixels": self.config.data.video_total_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        batch_dict = collate_fn(items)
        monitor_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        monitor_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(monitor_batch.batch))], dtype=object
        )

        gen_batch = monitor_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "has_offline_trajectory",
                "offline_output",
            ],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_total_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )

        world_size = self.actor_rollout_ref_wg.world_size
        gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        if pad_size > 0:
            gen_batch_output = unpad_dataproto(gen_batch_output, pad_size * n)
        monitor_batch = monitor_batch.repeat(repeat_times=n, interleave=True)
        monitor_batch = monitor_batch.union(gen_batch_output)

        reward_batch = monitor_batch.select(
            batch_keys=["responses", "response_mask"],
            non_tensor_batch_keys=[
                k for k in monitor_batch.non_tensor_batch if k != "multi_modal_data"
            ],
        )
        reward_tensor, reward_metrics = ray.get(
            self.reward_fn.compute_reward.remote(reward_batch)
        )

        accuracy_scores = reward_metrics.get("accuracy", [])
        uids = monitor_batch.non_tensor_batch["uid"]
        uid2acc = defaultdict(list)
        for uid, acc in zip(uids, accuracy_scores):
            uid2acc[uid].append(acc)

        cohort_size = len(uid2acc)
        all_correct = sum(
            1 for acc_list in uid2acc.values() if all(a >= 1.0 for a in acc_list)
        )
        retention_rate = all_correct / cohort_size if cohort_size > 0 else 0.0

        record = {
            "global_step": self.global_step,
            "cohort_size": cohort_size,
            "all_correct": all_correct,
            "retention_rate": round(retention_rate, 4),
        }
        self.monitor_log.append(record)

        metrics["monitor/cohort_size"] = cohort_size
        metrics["monitor/all_correct"] = all_correct
        metrics["monitor/retention_rate"] = retention_rate

        print(
            f"[Monitor] cohort_size={cohort_size}, all_correct={all_correct}, "
            f"retention={retention_rate:.2%}"
        )

    # ==================== End Review Queue ====================

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        timing_raw = {}
        total_gen_time = 0.0  # 累加生成时间
        total_reward_time = 0.0  # 累加奖励计算时间
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_total_pixels": self.config.data.video_total_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # V3 Review Queue: replace tail of batch with review samples before rollout
            if self.config.algorithm.review_queue_enabled:
                new_batch = self._prepare_review_replacement(new_batch, metrics)

            # pop those keys for generation
            # Mix-policy: 同时传递预采集轨迹信息 (has_offline_trajectory, offline_output)
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "multi_modal_data",
                    "has_offline_trajectory",  # Mix-policy: 是否有预采集轨迹
                    "offline_output",  # Mix-policy: 预采集的输出文本
                ],
                meta_info_keys=[
                    "image_min_pixels",
                    "image_max_pixels",
                    "video_min_pixels",
                    "video_max_pixels",
                    "video_total_pixels",
                    "video_fps",
                    "video_max_frames",
                ],
            )

            self._assert_multimodal_contract(gen_batch, stage="train")
            batch_has_offline = gen_batch.non_tensor_batch.get("has_offline_trajectory", None)
            batch_offline_output = gen_batch.non_tensor_batch.get("offline_output", None)

            # generate a batch
            t_start = time.time()
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            total_gen_time += time.time() - t_start

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                remax_reward_batch = new_batch.select(
                    batch_keys=["responses", "response_mask"],
                    non_tensor_batch_keys=[k for k in new_batch.non_tensor_batch if k != "multi_modal_data"],
                )
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(remax_reward_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            mix_policy_acc_threshold = getattr(self.config.worker.rollout, "mix_policy_accuracy_threshold", None)
            cached_filter_reward = None
            has_any_offline = batch_has_offline is not None and bool(np.any(np.asarray(batch_has_offline, dtype=bool)))
            if (
                self.config.worker.rollout.enable_mix_policy
                and mix_policy_acc_threshold is not None
                and self.config.worker.rollout.n > 1
                and has_any_offline
            ):
                if not (0.0 <= mix_policy_acc_threshold <= 1.0):
                    raise ValueError(
                        "worker.rollout.mix_policy_accuracy_threshold must be in [0, 1]. "
                        f"Got {mix_policy_acc_threshold}."
                    )

                # Compute reward on pure online rollouts first, then decide whether to inject offline per group.
                online_eval_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                online_eval_batch = online_eval_batch.union(gen_batch_output)
                online_reward_batch = online_eval_batch.select(
                    batch_keys=["responses", "response_mask"],
                    non_tensor_batch_keys=[k for k in online_eval_batch.non_tensor_batch if k != "multi_modal_data"],
                )
                t_start = time.time()
                online_reward_tensor, online_reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(online_reward_batch)
                )
                total_reward_time += time.time() - t_start

                if "accuracy" not in online_reward_metrics:
                    raise KeyError(
                        "Conditional mix-policy requires reward metric `accuracy`, but it was not returned by reward_fn."
                    )

                online_uids = online_eval_batch.non_tensor_batch["uid"]
                uid2acc = defaultdict(list)
                for uid, acc in zip(online_uids, online_reward_metrics["accuracy"]):
                    uid2acc[uid].append(0.0 if acc is None else float(acc))
                uid2mean_acc = {uid: float(np.mean(accs)) for uid, accs in uid2acc.items()}

                original_uids = new_batch.non_tensor_batch["uid"]
                replace_offline_mask = np.zeros(len(original_uids), dtype=bool)
                for i, uid in enumerate(original_uids):
                    if not bool(batch_has_offline[i]):
                        continue
                    if uid2mean_acc.get(uid, 1.0) < mix_policy_acc_threshold:
                        replace_offline_mask[i] = True

                replaced_groups = self._mix_offline_trajectories_in_gen_output(
                    gen_batch_output=gen_batch_output,
                    batch_has_offline=batch_has_offline,
                    batch_offline_output=batch_offline_output,
                    replace_offline_mask=replace_offline_mask,
                    n=self.config.worker.rollout.n,
                )
                metrics["mix_policy/acc_threshold"] = float(mix_policy_acc_threshold)
                metrics["mix_policy/online_groups_evaluated"] = len(original_uids)
                metrics["mix_policy/offline_candidates"] = int(np.sum(batch_has_offline.astype(bool)))
                metrics["mix_policy/offline_triggered_groups"] = int(np.sum(replace_offline_mask))
                metrics["mix_policy/offline_replaced_groups"] = int(replaced_groups)

                if self.config.algorithm.online_filtering and replaced_groups == 0:
                    # No response changed, so online reward can be reused for filtering.
                    cached_filter_reward = (online_reward_tensor, online_reward_metrics)

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                if cached_filter_reward is None:
                    t_start = time.time()
                    filter_reward_batch = new_batch.select(
                        batch_keys=["responses", "response_mask"],
                        non_tensor_batch_keys=[k for k in new_batch.non_tensor_batch if k != "multi_modal_data"],
                    )
                    reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(filter_reward_batch))
                    total_reward_time += time.time() - t_start
                else:
                    reward_tensor, reward_metrics = cached_filter_reward
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                # V3: process review results before filtering (sees all samples)
                if self.config.algorithm.review_queue_enabled:
                    self._process_review_results(reward_metrics, new_batch, metrics)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                    # 记录细粒度计时到 metrics
                    timing_raw["rollout_generate_part"] = total_gen_time
                    timing_raw["reward_compute_part"] = total_reward_time
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        # Only pass fields needed by reward, excluding large multi_modal_data to reduce serialization
                        reward_batch = batch.select(
                            batch_keys=["responses", "response_mask"],
                            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                        )
                        reward_ref = self.reward_fn.compute_reward.remote(reward_batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor

                        # V3: process review results with raw reward_metrics
                        if self.config.algorithm.review_queue_enabled:
                            self._process_review_results(reward_metrics, batch, metrics)

                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                # V3: evaluate monitor cohort on review schedule
                if (
                    self.config.algorithm.review_queue_enabled
                    and self.config.algorithm.monitor_cohort_enabled
                    and self._is_review_step()
                    and not self.monitor_collecting
                    and len(self.monitor_pids) > 0
                ):
                    with timer("monitor_eval", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        self._evaluate_monitor_cohort(metrics)
                        self.actor_rollout_ref_wg.release_rollout_engine()

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
