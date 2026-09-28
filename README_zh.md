# ReMind：RLVR 中的正确集合周转现象

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2606.03087)
[![Data](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/iieycx/rlsd-train-MMFineReason-123K)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

**ReMind** 研究可验证奖励强化学习（RLVR）中的正确集合周转现象，并提出一种保留感知的复习机制。该方法追踪模型已经解对的 prompt，并在 rollout 前周期性复用少量已掌握样本，以降低模型对已解样本的回退，同时避免额外的 rollout 轮次。

本仓库提供 ReMind 训练实现、vanilla GRPO 对照配置、图文训练入口和 checkpoint 转换工具。

<p align="center">
  <img src="assets/figure1_intro.png" width="90%">
</p>

<p align="center">
  <img src="assets/method_overview.png" width="95%">
</p>

## 概览

在 RLVR 训练中，模型可能在某个阶段解对一个 prompt，但在后续训练中又对同一 prompt 失败。ReMind 不再把每个 batch 完全视为独立样本，而是维护一个轻量级复习池，将未来训练 batch 中的一小部分样本替换为已掌握样本。

方法目标包括：

- 度量 RLVR 训练中的正确集合周转现象；
- 通过 rollout 前的复习替换保留已解样本；
- 避免为复习样本额外增加 rollout 轮次；
- 支持基于已发布数据集的多模态推理实验。

算法细节和实验结果请参考论文：

- [Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR](https://arxiv.org/abs/2606.03087)

## 数据

训练和验证数据已发布在 Hugging Face：

- [iieycx/rlsd-train-MMFineReason-123K](https://huggingface.co/datasets/iieycx/rlsd-train-MMFineReason-123K)

可以通过以下命令下载：

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='iieycx/rlsd-train-MMFineReason-123K',
    repo_type='dataset',
    local_dir='./data'
)
"
```

数据集包含：

- `MMFineReason_data_with_conclusion.json`：训练样本；
- `MMFineReason_images/`：对应的训练图片；
- `vl_math_val_mini_data.json`：验证样本；
- `val_data_image.zip`：验证图片。

## 安装

创建 Python 环境：

```bash
conda create -n remind python=3.11
conda activate remind
```

克隆仓库并安装训练依赖：

```bash
git clone https://github.com/cyuQ1n/Correct-Set-Turnover-in-RLVR.git
cd Correct-Set-Turnover-in-RLVR
pip install -e .
pip install flash-attn --no-build-isolation
```

训练需要 Linux、NVIDIA GPU 和兼容的 CUDA 环境。依赖使用 PyTorch 2.8.0、vLLM 0.11.0 和 Transformers 4.57.3；默认 padding-free 配置需要 FlashAttention。模型默认为 `Qwen/Qwen3-VL-8B-Instruct`，也可指定本地模型目录。

## 训练

下载数据后，将 `val_data_image.zip` 解压到数据目录，确保 JSON 中的相对图片路径能在 `IMAGE_DIR` 下找到。训练数据使用 `problem`、`answer`、`images`、`problem_type` 等字段。`problem_id` 应唯一且非空；未提供时，加载器会在过滤前按行号生成稳定 ID。

```bash
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export TRAIN_DATA=/absolute/path/to/data/MMFineReason_data_with_conclusion.json
export VAL_DATA=/absolute/path/to/data/vl_math_val_mini_data.json
export IMAGE_DIR=/absolute/path/to/data

# Vanilla GRPO 对照
bash examples/remind/run_grpo.sh

# GRPO + ReMind
bash examples/remind/run_grpo_remind.sh
```

两个命令分别启动一次训练，输出写入不同的实验目录。先加上 `--dry-run` 可查看完整命令，不加载模型或启动训练。命令末尾可追加 OmegaConf 参数，例如 `trainer.max_steps=10 trainer.val_before_train=false`。默认日志写入终端和本地文件；使用 W&B 时先运行 `wandb login`，再追加 `'trainer.logger=[console,file,wandb]'`。

| 设置 | 默认值 |
|---|---|
| 每步 prompt 数 / 每个 prompt 的 rollout 数 | 256 / 8 |
| 训练轮数 / 学习率 | 1 / 1e-6 |
| prompt / response 最大长度 | 4096 / 4096 |
| 每次复习的替换比例 / 复习间隔 | 0.10 / 5 步 |
| 入队概率 / 队列容量 | 0.25 / 20,000 |
| 开始复习的步数 | 50 |

默认使用 1 节点、8 GPU，rollout tensor parallel size 为 4。可通过 `NPROC_PER_NODE` 和 `worker.rollout.tensor_parallel_size=...` 调整硬件配置，并相应检查 batch 的可整除性。多节点训练需先启动 Ray 集群，在各节点准备相同的环境和文件，再在主节点设置 `RAY_ADDRESS`、`NNODES` 并运行一次启动命令。

ReMind 从标准训练流中将全对样本概率性入队，在复习步中用队列样本替换 batch 尾部，使用当前模型重新生成 rollout。复习全对的样本毕业退出队列，退化的样本重新入队。默认复习约占训练 prompt 的 2%，且不增加每步 rollout 数。独立监测 cohort 默认关闭；`algorithm.monitor_cohort_enabled=true` 会额外生成用于分析的 rollout。

## 工具

顶层 `scripts/` 目录包含 checkpoint 工具。例如，可以将 FSDP sharded checkpoint 合并为 Hugging Face 模型目录：

```bash
CKPT_DIR=/path/to/checkpoints/experiment/global_step_xxx/actor \
DST_MODEL_DIR=/path/to/output/merged_model \
bash scripts/model_merger.sh
```

## 仓库结构

```text
├── assets/              # 论文和 README 使用的图片
├── examples/remind/     # GRPO / ReMind 入口、配置、prompt 和 reward
├── verl/                # 训练框架与复习队列实现
├── tests/               # CPU 单元测试和入口检查
├── scripts/             # Checkpoint 转换工具
├── requirements.txt     # 依赖参考
├── pyproject.toml       # 格式化和构建配置
├── setup.py             # Editable install 入口
└── LICENSE              # Apache-2.0 license
```

关键实现见 `verl/trainer/ray_trainer.py` 和 `verl/trainer/config.py`。`make test` 运行 CPU 单元测试，覆盖复习队列行为、配置和启动命令；完整模型训练需要上述 GPU 环境。

## 引用

```bibtex
@misc{qin2026learningsolveforgettingretain,
      title={Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR},
      author={Chuanyu Qin and Chenxu Yang and Qingyi Si and Naibin Gu and Peng Fu and Zheng Lin},
      year={2026},
      eprint={2606.03087},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03087},
}
```

## 致谢

本项目基于以下开源项目构建：

- [EasyVideoR1](https://github.com/cyuQ1n/EasyVideoR1)：Easier RL for video understanding
- [EasyR1](https://github.com/hiyouga/EasyR1)：高效可扩展的多模态 RL 训练框架
- [veRL](https://github.com/volcengine/verl)：高性能 RL 与 HybridEngine

## License

本仓库使用 [Apache License 2.0](LICENSE) 开源。
