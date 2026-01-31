# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

SGT 是一个生成式推荐系统学习与复现项目，包含两个主要部分：

1. **MiniOneRec/** - 生成式推荐框架（SID 构建 + SFT + RL）
2. **step1/toy_rec/** - PyTorch 入门教学项目

## 技术栈

- Python 3.11 + PyTorch 2.6
- Transformers + TRL（GRPO 强化学习）
- DeepSpeed ZeRO-2 + Accelerate（分布式训练）
- 混合精度训练（bf16）

## 常用命令

### 环境设置

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
pip install -r MiniOneRec/requirements.txt
```

### toy_rec 训练（Step 1 学习）

```powershell
# 内置 toy 数据集
python -m toy_rec.train --dataset toy --num-items 200 --max-len 50 --epochs 3 --device cuda --amp

# JSONL 数据集
python toy_rec/make_toy_data.py --out ./toy_rec/data/toy.jsonl --num-users 2000 --num-items 500 --max-len 50 --n 20000
python -m toy_rec.train --dataset jsonl --train-path ./toy_rec/data/toy.jsonl --num-items 500 --max-len 50 --epochs 3 --device cuda --amp
```

### MiniOneRec 完整流程

```bash
# 1. 数据预处理
bash data/amazon18_data_process.sh --dataset Industrial_and_Scientific --user_k 5 --item_k 5

# 2. 文本转嵌入
bash rq/text2emb/amazon_text2emb.sh --dataset Industrial_and_Scientific --plm_name qwen

# 3. SID 构建（选一种）
bash rq/rqvae.sh              # RQ-VAE
bash rq/rqkmeans_constrained.sh  # 约束 RQ-Kmeans
bash rq/rqkmeans_plus.sh      # RQ-Kmeans+

# 4. 数据格式转换
python convert_dataset.py --dataset_name Industrial_and_Scientific --data_dir /path/to/data --output_dir /path/to/output

# 5. SFT 训练
bash sft.sh

# 6. RL 训练
bash rl.sh

# 7. 评估
bash evaluate.sh
```

### 单独运行训练脚本

```bash
# SFT（8 GPU）
torchrun --nproc_per_node 8 sft.py \
    --base_model <model_path> \
    --batch_size 1024 \
    --micro_batch_size 16 \
    --train_file <train_file> \
    --eval_file <eval_file> \
    --output_dir <output_dir>

# RL（DeepSpeed ZeRO-2）
accelerate launch --config_file ./config/zero2_opt.yaml --num_processes 8 rl.py \
    --model_path <model_path> \
    --train_batch_size 64 \
    --num_train_epochs 2 \
    --reward_type ranking \
    --num_generations 16
```

## 代码架构

### MiniOneRec 三阶段流程

```
物品文本 → 文本编码器 → 嵌入向量 → RQ-VAE/RQ-Kmeans → SID（语义 token）
                                                          ↓
用户历史序列 → SFT（下一物品预测 + 语言对齐）→ RL（GRPO + 约束解码）→ 推荐结果
```

### 核心模块

| 文件 | 功能 |
|------|------|
| `sft.py` / `sft_gpr.py` | SFT 训练（标准 / GPR 启发） |
| `rl.py` / `rl_gpr.py` | RL 训练（GRPO / HEPO） |
| `minionerec_trainer.py` | GRPO 训练器核心实现 |
| `data.py` | 数据管道（10+ 种数据集类） |
| `LogitProcessor.py` | 约束解码（保证生成合法 SID） |
| `evaluate.py` | 离线评估（HR@K, NDCG@K） |
| `rq/` | SID 构建模块（RQ-VAE, RQ-Kmeans 等） |

### toy_rec 模块（学习用）

| 文件 | 功能 |
|------|------|
| `dataset.py` | 数据集 + collate_fn（padding/mask） |
| `model.py` | 简单推荐模型（Embedding + Pooling + Linear） |
| `train.py` | 训练入口（AMP、梯度累积、评估） |
| `metrics.py` | HR@K, NDCG@K 指标计算 |

## 数据目录结构

```
data/Amazon/
├── train/          # 训练集 CSV
├── valid/          # 验证集 CSV
├── test/           # 测试集 CSV
├── info/           # 物品元数据
└── index/
    ├── *.index.json      # SID 映射表
    ├── *.item.json       # 物品详细信息
    └── *.emb-qwen-td.npy # 预计算嵌入
```

## 关键概念

- **SID (Semantic ID)**: 将物品转换为紧凑的语义 token，通过 RQ-VAE 或 RQ-Kmeans 构建
- **约束解码**: LogitProcessor 保证生成的每个 token 都是有效 SID
- **GRPO**: Group Relative Policy Optimization，推荐专用强化学习算法
- **形状约定**: B=batch_size, T=max_len, V=num_items+1, D=embed_dim

## 注意事项

- 使用 Instruct 模型时，如果评估日志中 CC 指标非零，说明约束解码未生效，建议切换到 base 模型
- RL 阶段可使用数万条样本的子集训练，边际收益递减
- 硬件需求：4-8 × A100/H100 80GB（完整流程）
