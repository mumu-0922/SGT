# toy_rec (Step1 mini project)
# This folder contains a minimal, reusable PyTorch training pipeline for
# sequence recommendation / next-item prediction (toy version).
# It is designed for Step 1 practice in plan/jihua.md and can be reused later.
# The code is heavily commented in Chinese for learning.
# Tip: run from repo root, e.g. `python -m toy_rec.train` (recommended).

<!--
BEGINNER GUIDE (Step 1) --------------------------------------------------------------------
This README is designed to be read in order. If you feel lost, do not jump around.
Goal: in 2 days you can run a full training loop, understand batch shapes, and save/load checkpoints.
File order: dataset.py -> model.py -> train.py -> metrics.py -> utils.py -> make_toy_data.py
Keywords: input_ids, attention_mask, labels, logits, loss, backward, optimizer.step, checkpoint.
If you only do one thing: print tensor shapes and confirm loss decreases.
-------------------------------------------------------------------------------------------
-->

## 新手学习路线（Step 1：Python / PyTorch）

### 你现在要达成的唯一目标（别分心）

两天内，把下面这条链路**跑通 + 看懂 + 能改**：

> 原始样本（变长序列）→ `DataLoader` 组 batch → `nn.Embedding` 编码 → `loss` → `backward()` → `optimizer.step()` → 保存/加载 checkpoint → 看指标

你现在不需要懂大模型、LoRA、RL。先把“训练闭环”练成肌肉记忆。

### Step1 学什么，对应 toy_rec 哪里？

| Step1 要学的东西 | 你在项目里会看到的样子 | 在 toy_rec 里对应 | 你要看懂的点 |
|---|---|---|---|
| `Dataset` / `DataLoader` / `collate_fn` | 一堆样本怎么变成一锅 batch | `dataset.py` 的 `SeqRecSample` / `collate_seqrec` | 变长序列如何 padding；mask 有什么用 |
| `nn.Embedding` | 离散 id → 向量 | `model.py` 的 `item_emb` | 为什么 0 用来 padding；embedding 也会被训练 |
| `nn.Module` / `forward` | 模型怎么吃 batch，吐 logits | `model.py` 的 `SimpleSeqRecModel.forward` | 输入/输出形状（B/T/V/D） |
| `CrossEntropyLoss` | next-item 训练目标 | `train.py` 的 `loss_fn = CrossEntropyLoss()` | logits 是“每个 item 的分数”；label 是 target item id |
| `backward()` / `optimizer.step()` | 参数更新 | `train.py` 的训练 loop | 四步：forward→loss→backward→step→zero_grad |
| checkpoint（断点） | 暂停/恢复训练，方便复现 | `utils.py` 的 `save_checkpoint/load_checkpoint` + `train.py` | 为什么要存 model+optimizer(+scaler) |
| 评估指标 | HR@K / NDCG@K | `metrics.py` 的 `hr_ndcg_at_k` + `train.py` 的 `evaluate()` | Top-K 是怎么来的；rank 越靠前分越高 |

> 形状速记（非常重要）：  
> - `B` = batch_size（一次喂多少条样本）  
> - `T` = max_len（每条历史序列的长度，短的会 padding）  
> - `V` = num_items+1（物品总数 + padding 0）  
> - `D` = embed_dim（向量维度）  

### 两天怎么学（照着做就行）

**Day 1（先学数据流）**

1. 先读 `dataset.py`：只盯 `SeqRecSample` 和 `collate_seqrec`
2. 运行一次训练（哪怕你看不懂也先跑）：  
   - `python -m toy_rec.train --dataset toy --num-items 200 --max-len 50 --epochs 1 --device cpu`
3. 加 3 行打印（建议你自己改）：在 `train.py` 里打印 `input_ids.shape / attention_mask.sum() / labels[:5]`
4. 验收：你能用一句话解释：**padding 是什么？mask 是什么？label 是什么？**

**Day 2（再学训练闭环）**

1. 读 `model.py`：只盯 `SimpleSeqRecModel.forward()`，把形状对上
2. 再读 `train.py`：找到训练 loop，按顺序理解：  
   - `out = model(batch)` → `loss_fn(logits, labels)` → `backward()` → `step()` → `zero_grad()`
3. 用 GPU 跑一遍（你有 CUDA 才用）：  
   - `python -m toy_rec.train --dataset toy --num-items 200 --max-len 50 --epochs 3 --device cuda --amp`
4. 验收：loss 能下降；你知道改 `batch_size/max_len/embed_dim` 会影响什么（速度/显存/效果）

### 阅读顺序（别跳着看）

1. `dataset.py`（样本→batch）
2. `model.py`（batch→logits）
3. `train.py`（logits→loss→更新→评估→保存）
4. `metrics.py`（指标怎么计算）
5. `utils.py`（seed/断点）
6. `make_toy_data.py`（最后再看：生成 JSONL toy 数据）

### 新手最常见问题（别卡死）

- 报错 `No module named torch`：说明你还没装 PyTorch（先装，再运行）
- `--device cuda` 但 `cuda is not available`：要么没装 CUDA 版 torch，要么驱动/CUDA 环境没配好；先用 `--device cpu` 跑通
- `Expected all tensors to be on the same device`：batch 在 CPU，model 在 GPU（或反过来）；检查 `.to(device)`
- shape 不匹配：先打印 `input_ids.shape / logits.shape / labels.shape`，再看哪里维度不对
- OOM（显存爆）：先把 `--batch-size` 调小，再开 `--amp`，再把 `--max-len` 调小，再开 `--grad-accum`

### 以后怎么复用到 MiniOneRec？

你只需要记住这套“接口”：

- dataset/collate 输出：`input_ids`、`attention_mask`、`labels`
- model 接收 batch，输出 `logits`
- train 用 `CrossEntropyLoss(logits, labels)` 做 SFT（下一物品预测）

MiniOneRec 会把 item 变成更复杂的 token（例如 SID/约束生成等），但**数据组织 + 训练闭环**和这里是一致的。

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
