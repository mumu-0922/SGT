# 28 天 LLM 学习笔记

---

# Step 1：Python / PyTorch 基础（2 天）

## Day 1：数据流

### 学了什么
- `dataset.py` 里的 `SeqRecSample` 和 `collate_seqrec`

### 核心概念

| 概念 | 含义 |
|------|------|
| **padding** | 用 0 填充短序列，让所有样本长度一样 |
| **mask** | 1 = 真实数据，0 = padding |
| **label** | 要预测的下一个物品 id |

### 关键代码

```python
# collate_seqrec 的输出
{
    "input_ids": [B, T],       # 填充后的序列
    "attention_mask": [B, T],  # 哪些是真实数据
    "labels": [B],             # 要预测的目标
}
```

### 实验结果

```
input_ids.shape = torch.Size([256, 50])   # 256条样本，每条长度50
attention_mask.sum() = 6867               # 6867个真实token
labels[:5] = tensor([97, 91, 153, 96, 165])  # 前5个目标
```

### 我的理解
- DataLoader 每次给一堆样本，长度不一样
- collate_fn 把它们 padding 成相同长度，拼成矩阵
- mask 告诉模型哪些是真的，哪些是填充的

---

## Day 2：模型 + 训练循环

### 学了什么
- `model.py` 里的 `SimpleSeqRecModel.forward()`
- `train.py` 里的训练循环

### 模型形状变化

```
input_ids   [B, T]        例如 [256, 50]
    ↓ Embedding（查表，id → 向量）
x           [B, T, D]     例如 [256, 50, 128]
    ↓ Pooling（取最后一个有效位置）
pooled      [B, D]        例如 [256, 128]
    ↓ Linear（矩阵乘法，打分）
logits      [B, V]        例如 [256, 201]
```

### 训练循环

```python
for batch in train_loader:
    out = model(batch)           # ① 前向传播
    loss = loss_fn(logits, labels)  # ② 算 loss
    loss.backward()              # ③ 反向传播（算梯度）
    optimizer.step()             # ④ 更新权重
    optimizer.zero_grad()        # ⑤ 清空梯度
```

### 核心概念

| 概念 | 含义 |
|------|------|
| **Embedding** | 把 id 变成向量（查表） |
| **Pooling** | 把序列变成一个向量（取最后一个） |
| **Linear** | 矩阵乘法，对每个物品打分 |
| **epoch** | 所有样本训练一遍 |
| **batch** | 一次训练的样本数（如 256） |

### 参数影响

| 参数 | 增大后 | 影响 |
|------|--------|------|
| `batch_size` | 256→512 | 显存↑，batch数↓ |
| `max_len` | 50→100 | 显存↑，计算量↑ |
| `embed_dim` | 128→256 | 显存↑，模型容量↑ |

### 实验结果

```
batch_size=256: train_batches=70
batch_size=512: train_batches=35（减半）

loss 变化：5.48 → 0.21 → 0.01（3个epoch）
```

### 我的理解
- 模型就是 Embedding + Pooling + Linear
- 训练就是不断调整权重，让 loss 变小
- batch_size 越大，显存占用越多，但训练越快

---

## 形状速记

- `B` = batch_size（一次喂多少条样本）
- `T` = max_len（序列长度）
- `V` = num_items + 1（物品总数 + padding 0）
- `D` = embed_dim（向量维度）
