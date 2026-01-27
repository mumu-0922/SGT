# toy_rec (Step1 mini project)
# This folder contains a minimal, reusable PyTorch training pipeline for
# sequence recommendation / next-item prediction (toy version).
# It is designed for Step 1 practice in plan/jihua.md and can be reused later.
# The code is heavily commented in Chinese for learning.

## 这是什么？

`toy_rec/` 是一个“最小可复用”的序列推荐训练骨架，目标是帮你在 **Step 1（Python/PyTorch）** 两天内把下面这条链路跑通并真正理解：

> 数据（变长序列） → `Dataset/DataLoader/collate_fn` → `nn.Embedding` → loss → `backward()` → 保存/加载 checkpoint → 评估指标

后面你复现 MiniOneRec 时，很多概念/代码结构可以直接迁移：数据 batch 的组织方式、mask/padding、训练 loop、评估与记录实验等。

## 目录结构

- `dataset.py`：Toy 数据集 + JSONL 数据集 + `collate_fn`（padding/mask）
- `model.py`：最小推荐模型（Embedding + pooling + Linear）
- `metrics.py`：HR@K、NDCG@K 等基础指标
- `utils.py`：seed、保存/加载 checkpoint、小工具
- `train.py`：训练入口（可选 AMP、梯度累积、评估、保存）
- `make_toy_data.py`：生成一个可学习的 toy JSONL 数据集（可选）

## 快速开始

1) 确保你有 Python 和 PyTorch（能 `import torch`）。

2) 直接用“内置 toy 数据集”跑通训练（最简单）：

```powershell
cd D:\AI\SGT
python .\toy_rec\train.py --dataset toy --num-items 200 --max-len 50 --epochs 3 --device cuda --amp
```

3)（可选）生成一个 JSONL toy 数据集，再用 JSONL 训练：

```powershell
python .\toy_rec\make_toy_data.py --out .\toy_rec\data\toy.jsonl --num-users 2000 --num-items 500 --max-len 50 --n 20000
python .\toy_rec\train.py --dataset jsonl --train-path .\toy_rec\data\toy.jsonl --num-items 500 --max-len 50 --epochs 3 --device cuda --amp
```

## 你应该能学会什么（Step 1 验收）

- 能解释：为什么需要 `collate_fn`、padding 和 `attention_mask`
- 能解释：`nn.Embedding` 在推荐里对应什么
- 能自己改：max_len/batch_size/lr，观察 loss 与 HR@K 的变化
- 能排查：shape/device/dtype 常见错误，和 OOM（显存爆）怎么缓解

