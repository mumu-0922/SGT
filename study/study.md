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

---

# Step 2：学 Transformer（2-3 天）

## Day 1：QKV 和注意力机制

### 学了什么
- Attention 的核心思想
- Q、K、V 的含义和计算

### 为什么需要 Attention？

Step 1 的 Last Pooling 只看最后一个位置，丢失了前面的信息。

Attention 让模型**自己决定**看哪些位置、看多少。

### QKV 含义

| 概念 | 含义 | 类比（图书馆找书） |
|------|------|-------------------|
| **Q (Query)** | 我要找什么 | 你的问题："找科幻小说" |
| **K (Key)** | 每个位置的标签 | 书架标签："科幻"、"言情" |
| **V (Value)** | 每个位置的内容 | 书的实际内容 |

### QKV 怎么来的？

从同一个输入 X，通过**不同的可学习矩阵**变换得到：

```
Q = X × W_Q
K = X × W_K
V = X × W_V
```

**为什么要乘 W？** 让相似度可学习，而不是固定的。

### Attention 计算流程

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

| 步骤 | 做什么 | 目的 |
|------|--------|------|
| Q × K^T | 算相似度 | 找出哪些位置相关 |
| ÷ √d | 缩放 | 防止数值太大 |
| softmax | 变成权重 | 权重和为 1 |
| × V | 加权求和 | 综合相关位置的信息 |

### softmax 计算

把任意数字变成概率（0~1，和为1）：

```
输入：[0.71, 0, 0.71]

e^0.71 = 2.03, e^0 = 1.00, e^0.71 = 2.03
总和 = 5.06

输出：[2.03/5.06, 1.00/5.06, 2.03/5.06] = [0.40, 0.20, 0.40]
```

### 我的理解
- Attention 让模型能"看到"所有位置，而不只是最后一个
- Q、K、V 通过可学习的 W 矩阵得到
- softmax 把相似度变成权重，权重大的位置贡献更多

---

## Day 2：多头注意力 + 完整 Transformer

### 学了什么
- 多头注意力（Multi-Head Attention）
- 残差连接（Residual Connection）
- 层归一化（LayerNorm）
- 前馈网络（FFN）
- 完整 Transformer Layer 结构

### 1. 多头注意力

**为什么要多头？** 一个头只能学一种关系，多个头能学多种关系。

```
单头：只能学"语法关系"
多头：头1学语法，头2学语义，头3学位置...
```

**怎么做？** 把 D 维度分成 H 份，每份独立做 Attention，最后拼回来。

```
输入 x:     [B, T, D]      例如 [256, 50, 128]
分成 H 头:  [B, T, H, d]   例如 [256, 50, 8, 16]  (D=128, H=8, d=16)
每头独立 Attention
拼回来:    [B, T, D]      例如 [256, 50, 128]
```

### 2. 残差连接

**公式：** `output = F(x) + x`

**作用：** 保留原始信息，防止深层网络信息丢失。

```python
attn_out = attention(x)
x = x + attn_out  # 残差：变换结果 + 原始输入
```

**类比：** 老师改作文，在原文基础上修改，而不是重写。

**注意：** 每层的 x 是上一层的输出，不是最初的输入（类似递归）。

```
Layer 1:  X₀ → F₁(X₀) + X₀ = X₁
Layer 2:  X₁ → F₂(X₁) + X₁ = X₂
Layer 3:  X₂ → F₃(X₂) + X₂ = X₃
```

### 3. LayerNorm（层归一化）

**公式：** `LayerNorm(x) = (x - 均值) / 标准差 × γ + β`

**作用：** 把数值标准化到均值=0、方差=1，防止数值爆炸/消失。

```
输入：[2, 4, 6, 8]
均值 = 5，标准差 ≈ 2.24
输出：[-1.34, -0.45, +0.45, +1.34]
```

### 4. FFN（前馈网络）

**公式：** `FFN(x) = Linear2(ReLU(Linear1(x)))`

**形状变化：**
```
x:       [B, T, D]    → [256, 50, 128]
Linear1: D → 4D       → [256, 50, 512]  （扩大 4 倍）
ReLU:    负数变 0     → [256, 50, 512]
Linear2: 4D → D       → [256, 50, 128]  （缩回来）
```

**作用：** Attention 负责"看关系"，FFN 负责"做变换/思考"。

### 5. 完整 Transformer Layer

```
输入 x: [B, T, D]
    ↓
① Attention（学位置关系）
    ↓
② + x（残差连接）
    ↓
③ LayerNorm（稳定数值）
    ↓
④ FFN（特征变换）
    ↓
⑤ + x（残差连接）
    ↓
⑥ LayerNorm（稳定数值）
    ↓
输出: [B, T, D]（形状不变）
```

**代码：**
```python
class TransformerLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # ①②③ Attention + 残差 + LayerNorm
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # ④⑤⑥ FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

### 6. 多层堆叠

整个 ①②③④⑤⑥ 重复 N 遍，Layer 2 的输入 = Layer 1 的输出。

| 模型 | 层数 N |
|------|--------|
| GPT-2 Small | 12 |
| GPT-3 | 96 |
| LLaMA-7B | 32 |

```python
layers = nn.ModuleList([TransformerLayer() for _ in range(12)])

x = embedding(input_ids)
for layer in layers:
    x = layer(x)  # 每层输出作为下一层输入
```

### 关键公式汇总

| 组件 | 公式 |
|------|------|
| Attention | softmax(Q×K^T/√d) × V |
| 残差 | output = F(x) + x |
| LayerNorm | (x - 均值) / 标准差 × γ + β |
| FFN | Linear2(ReLU(Linear1(x))) |

### 我的理解
- 多头让模型从多个角度看关系
- 残差保证信息不丢失，能训练深层网络
- LayerNorm 稳定数值，防止爆炸/消失
- FFN 在每个位置独立做特征变换
- 整个流程重复 N 层，逐层提取更高级的特征

---

## Day 3：RoPE 旋转位置编码

（待学习）

---

## 形状速记

- `B` = batch_size（一次喂多少条样本）
- `T` = max_len（序列长度）
- `V` = num_items + 1（物品总数 + padding 0）
- `D` = embed_dim（向量维度）
- `H` = num_heads（注意力头数）
