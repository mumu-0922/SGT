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

## Day 3：手写 Transformer（面试重点）

### 面试考察优先级

| 优先级 | 内容 | 考察点 |
|--------|------|--------|
| ⭐⭐⭐ | **Scaled Dot-Product Attention** | `softmax(QK^T / √d_k) × V` |
| ⭐⭐⭐ | **Multi-Head Attention** | 拆头、并行计算、拼接 |
| ⭐⭐ | **Position Encoding** | 正弦公式或 RoPE |
| ⭐⭐ | **Mask 机制** | padding mask、causal mask |

### 核心代码（背下来）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Scaled Dot-Product Attention（最核心）
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

# 2. Multi-Head Attention（第二重要）
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(model_dim, model_dim)
        self.W_V = nn.Linear(model_dim, model_dim)
        self.W_O = nn.Linear(model_dim, model_dim)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # 线性变换 + 拆头
        Q = self.W_Q(Q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(K).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(V).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Attention
        out = attention(Q, K, V, mask.unsqueeze(1) if mask is not None else None)
        # 拼接 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        return self.W_O(out)
```

### 形状变化（必须清楚）

```
输入 x: (B, T, D)           例如 (2, 5, 4)
    ↓ W_Q, W_K, W_V
Q, K, V: (B, T, D)          (2, 5, 4)
    ↓ view + transpose (拆头)
Q, K, V: (B, H, T, d)       (2, 2, 5, 2)  H=2, d=D/H=2
    ↓ Q × K^T / √d_k
scores: (B, H, T, T)        (2, 2, 5, 5)
    ↓ mask + softmax
weights: (B, H, T, T)       (2, 2, 5, 5)
    ↓ × V
out: (B, H, T, d)           (2, 2, 5, 2)
    ↓ transpose + view (拼接)
out: (B, T, D)              (2, 5, 4)
    ↓ W_O
output: (B, T, D)           (2, 5, 4)
```

### 常见追问

| 问题 | 答案 |
|------|------|
| 为什么除以 √d_k？ | 防止点积过大，softmax 梯度消失 |
| 多头的作用？ | 并行学习多种注意力模式 |
| Self-Attention vs Cross-Attention？ | Q=K=V 自己 vs Q 来自 Decoder，KV 来自 Encoder |
| 为什么用 LayerNorm 不用 BatchNorm？ | 序列长度不固定，LayerNorm 对每个位置独立归一化 |
| Encoder 和 Decoder 的区别？ | Decoder 多一个 Cross-Attention，且 Self-Attention 有 causal mask |

### 三种 Mask

| Mask | 用途 | 特点 |
|------|------|------|
| Padding Mask | 屏蔽填充位置 | 1=有效，0=padding |
| Causal Mask | 屏蔽未来位置 | 下三角矩阵 |
| Cross-Attention Mask | 屏蔽双方 padding | Decoder×Encoder |

### 面试技巧

1. 先写 `attention()` 函数（5行核心）
2. 再写 `MultiHeadAttention` 类
3. 边写边说形状变化
4. 被问到不会的，诚实说"这个我不太确定"

---

## Day 4：RoPE 旋转位置编码

### 学了什么
- RoPE（Rotary Position Embedding）旋转位置编码
- 主流大模型（LLaMA、Qwen、ChatGLM）使用的位置编码方式

### 正弦编码 vs RoPE

| 对比 | 正弦编码 | RoPE |
|------|----------|------|
| 方式 | 加法：`x + PE` | 旋转：`rotate(x, θ)` |
| 位置类型 | 绝对位置 | **相对位置** |
| 应用对象 | Embedding | **Q 和 K** |
| 使用模型 | 原版 Transformer | LLaMA, Qwen, ChatGLM |

### 核心思想

```
位置 m 的 Q 旋转 m*θ
位置 n 的 K 旋转 n*θ
点积 Q·K 自然包含相对位置 (m-n)
```

**关键洞察：** 两个旋转后向量的点积，只和它们的**相对位置差**有关。

### 2D 旋转公式

把向量 [x, y] 旋转 θ 角度：

```
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]

即：
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```

### 高维实现

把 D 维向量分成 D/2 组，每组 2 维，分别旋转：

```
向量 [x0, x1, x2, x3] (D=4)
     ↓
分组 [x0, x1] 和 [x2, x3]  (D/2=2 组)
     ↓
每组用不同频率旋转
```

频率公式：`θ_i = 1 / 10000^(2i/D)`

### 核心代码

```python
# 1. 预计算频率和角度
def precompute_rope_freqs(dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, dim/2)
    return torch.cos(angles), torch.sin(angles)

# 2. 应用旋转
def apply_rope(x, cos, sin):
    x1 = x[..., :dim//2]  # 前半
    x2 = x[..., dim//2:]  # 后半
    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos
    return torch.cat([x1_new, x2_new], dim=-1)

# 3. 在 Attention 中使用（只对 Q 和 K）
Q = apply_rope(Q, cos, sin)
K = apply_rope(K, cos, sin)
# V 不需要旋转！
```

### 为什么 V 不需要 RoPE？

- RoPE 的目的是让 Q·K 的点积包含相对位置信息
- V 是被加权求和的内容，不参与位置计算
- 位置信息已经通过 attention weights 传递了

### RoPE 的优势

| 优势 | 说明 |
|------|------|
| 相对位置 | 注意力分数只和距离有关，更符合直觉 |
| 外推性好 | 能处理比训练时更长的序列 |
| 计算高效 | 只需要逐元素乘法，无额外参数 |

### 面试要点

1. **核心思想**：旋转编码相对位置，Q·K 点积自然包含位置差
2. **实现方式**：分成 D/2 组，每组 2D 旋转，不同频率
3. **应用对象**：只对 Q 和 K，V 不需要
4. **优势**：相对位置、外推性好、计算高效

### 我的理解

- 正弦编码是"加"位置，RoPE 是"旋转"位置
- 旋转的巧妙之处：点积结果只和相对距离有关
- 主流大模型都用 RoPE，面试可能会问

---

## 形状速记

- `B` = batch_size（一次喂多少条样本）
- `T` = max_len（序列长度）
- `V` = num_items + 1（物品总数 + padding 0）
- `D` = embed_dim（向量维度）
- `H` = num_heads（注意力头数）
