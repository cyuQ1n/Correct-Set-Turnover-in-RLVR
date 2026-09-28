# Copyright 2026 ReMind authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real queue methods on CPU without launching Ray workers."""

import unittest
from collections import deque
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.config import AlgorithmConfig, PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.dataset import RLHFDataset, collate_fn


def item(pid):
    return {"input_ids": torch.tensor([pid, 0]), "problem_id": pid}


class PromptDataset:
    def get_items_by_problem_ids(self, ids):
        return [item(pid) for pid in ids]


def trainer(ids=(), ratio=0.5, capacity=20):
    result = RayPPOTrainer.__new__(RayPPOTrainer)
    result.config = PPOConfig()
    result.config.algorithm = AlgorithmConfig(
        review_queue_enabled=True,
        review_replace_ratio=ratio,
        review_queue_max_size=capacity,
        review_queue_sample_rate=1.0,
    )
    result.global_step = 50
    result.review_queue = deque(ids, maxlen=capacity)
    result.review_queue_set = set(ids)
    result.train_dataset = PromptDataset()
    result._current_review_uids = set()
    result._current_review_pid_by_uid = {}
    result.monitor_pid_set = set()
    result.monitor_collecting = False
    result.review_log = []
    return result


def batch(ids):
    result = DataProto.from_single_dict(collate_fn([item(pid) for pid in ids]))
    result.non_tensor_batch["uid"] = np.array([f"fresh-{i}" for i in ids], dtype=object)
    return result


class ReviewQueueTests(unittest.TestCase):
    def test_review_schedule(self):
        t = trainer()
        for step, expected in [(49, False), (50, True), (51, False), (55, True)]:
            t.global_step = step
            self.assertEqual(t._is_review_step(), expected)
        t.config.algorithm.review_queue_enabled = False
        self.assertFalse(t._is_review_step())

    def test_zero_ratio_does_not_dequeue(self):
        t = trainer([10, 11], ratio=0)
        original = batch([0, 1, 2, 3])
        self.assertIs(t._prepare_review_replacement(original, {}), original)
        self.assertEqual(list(t.review_queue), [10, 11])

    def test_replacement_uses_actual_batch_and_keeps_rollout_budget(self):
        t = trainer([10, 11, 12, 13])
        # Configured batch size is deliberately larger than the actual mini-batch.
        t.config.data.rollout_batch_size = 256
        result = t._prepare_review_replacement(batch([0, 1, 2, 3]), {})
        self.assertEqual(result.non_tensor_batch["problem_id"].tolist(), [0, 1, 10, 11])
        self.assertEqual(len(result.repeat(8, interleave=True)), 32)
        self.assertEqual(list(t.review_queue), [12, 13])
        self.assertEqual(t.review_queue_set, {12, 13})

    def test_all_correct_graduates_and_mixed_outcomes_return(self):
        t = trainer([10, 11, 12])
        result = t._prepare_review_replacement(batch([0, 1, 2, 3]), {})
        repeated = result.repeat(2, interleave=True)
        metrics = {}
        # Fresh 0 mastered, fresh 1 mixed; review 10 mastered, review 11 mixed.
        t._process_review_results({"accuracy": [1, 1, 1, 0, 1, 1, 0, 1]}, repeated, metrics)
        self.assertEqual(list(t.review_queue), [12, 11, 0])
        self.assertEqual(t.review_queue_set, {12, 11, 0})
        self.assertEqual(metrics["review/graduated_count"], 1)
        self.assertEqual(metrics["review/regressed_count"], 1)
        self.assertEqual(metrics["review/retention_rate"], 0.5)

    def test_regressed_prompts_keep_fifo_order(self):
        t = trainer([10, 11, 12, 13], ratio=1)
        result = t._prepare_review_replacement(batch([0, 1, 2]), {})
        t._process_review_results({"accuracy": [0, 1] * 3}, result.repeat(2, interleave=True), {})
        self.assertEqual(list(t.review_queue), [13, 10, 11, 12])

    def test_capacity_eviction_keeps_membership_consistent(self):
        t = trainer([10, 11], capacity=2)
        t._process_review_results({"accuracy": [1, 1]}, batch([0]).repeat(2, interleave=True), {})
        self.assertEqual(list(t.review_queue), [11, 0])
        self.assertEqual(t.review_queue_set, {11, 0})

    def test_sampling_probability_zero_prevents_new_enqueues(self):
        t = trainer()
        t.config.algorithm.review_queue_sample_rate = 0
        t._process_review_results({"accuracy": [1, 1]}, batch([0]).repeat(2, interleave=True), {})
        self.assertEqual(list(t.review_queue), [])


class DataAndConfigTests(unittest.TestCase):
    def test_generated_prompt_ids(self):
        source = Dataset.from_list([{"problem": "one", "answer": "1"}, {"problem": "two", "answer": "2"}])
        with patch("verl.utils.dataset.load_dataset", return_value=source):
            data = RLHFDataset("test-dataset", None, None, filter_overlong_prompts=False)
        self.assertEqual(list(data.dataset["problem_id"]), [0, 1])
        self.assertEqual(data._problem_id_to_idx, {0: 0, 1: 1})

    def test_duplicate_prompt_ids_fail_explicitly(self):
        data = RLHFDataset.__new__(RLHFDataset)
        data.dataset = Dataset.from_list([{"problem_id": "same"}, {"problem_id": "same"}])
        with self.assertRaisesRegex(ValueError, "unique"):
            data._build_problem_id_index()

    def test_configs_share_backbone_and_disable_extra_audits(self):
        configs = []
        for name in ["vanilla_grpo", "grpo_remind"]:
            c = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(f"examples/remind/configs/{name}.yaml"))
            self.assertFalse(c.algorithm.monitor_cohort_enabled)
            self.assertEqual(c.worker.rollout.n, 8)
            self.assertEqual(c.worker.actor.global_batch_size, 256)
            self.assertEqual(c.trainer.total_epochs, 1)
            configs.append(c)
        self.assertFalse(configs[0].algorithm.review_queue_enabled)
        self.assertTrue(configs[1].algorithm.review_queue_enabled)
        self.assertEqual(configs[0].worker, configs[1].worker)

    def test_invalid_review_settings(self):
        for kwargs in [{"review_freq": 0}, {"review_queue_max_size": 0}, {"review_replace_ratio": -0.1}, {"review_queue_sample_rate": 2}]:
            with self.assertRaises(ValueError):
                AlgorithmConfig(**kwargs).post_init()


if __name__ == "__main__":
    unittest.main()
