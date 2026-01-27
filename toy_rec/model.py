# -*- coding: utf-8 -*-
"""
toy_rec.model

最小推荐模型（Step1 够用版）：
- 输入：item id 序列（padding 后的 [B, T]）
- 编码：nn.Embedding 把离散 id 变成向量
- pooling：拿最后一个有效位置（或 mean pooling）
- 输出：线性层输出对所有 item 的 logits（分类问题）

这就是“下一物品预测”的最小闭环：NTP 的推荐版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch import nn


PoolingType = Literal["last", "mean"]


@dataclass
class ModelOutput:
    logits: torch.Tensor  # [B, V]


class SimpleSeqRecModel(nn.Module):
    """
    SimpleSeqRecModel

    为什么用这个模型？
    - 结构极简：embedding + pooling + linear
    - 你能把注意力集中在“训练链路/数据形状/损失/反传”上
    - 后面要替换成 Transformer/LLM，只要保持 batch 字段一致即可
    """

    def __init__(
        self,
        *,
        num_items: int,
        embed_dim: int = 128,
        pad_id: int = 0,
        pooling: PoolingType = "last",
    ) -> None:
        super().__init__()

        self.num_items = int(num_items)
        self.pad_id = int(pad_id)
        self.pooling: PoolingType = pooling

        # vocab 大小 = num_items + 1（0 作为 padding）
        self.item_emb = nn.Embedding(self.num_items + 1, embed_dim, padding_idx=self.pad_id)
        self.out = nn.Linear(embed_dim, self.num_items + 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """
        batch 约定来自 dataset.collate_seqrec：
        - input_ids: [B, T]
        - attention_mask: [B, T]（1 有效，0 padding）
        - lengths: [B]（可选）
        """

        input_ids = batch["input_ids"]  # [B, T]
        attention_mask = batch["attention_mask"]  # [B, T]

        # 1) Embedding：把 id → 向量
        x = self.item_emb(input_ids)  # [B, T, D]

        # 2) pooling：把序列 [B, T, D] 聚合成 [B, D]
        if self.pooling == "last":
            # last pooling：取最后一个有效位置（最常见的 next-item）
            lengths = batch.get("lengths")
            if lengths is None:
                lengths = attention_mask.sum(dim=1)
            # lengths 是“有效 token 数”，最后一个 token 的 index = lengths - 1
            last_index = torch.clamp(lengths - 1, min=0).view(-1, 1, 1)  # [B,1,1]
            last_index = last_index.expand(-1, 1, x.size(-1))  # [B,1,D]
            pooled = x.gather(dim=1, index=last_index).squeeze(1)  # [B,D]
        elif self.pooling == "mean":
            # mean pooling：对有效位置做均值（mask 很关键）
            mask = attention_mask.unsqueeze(-1).to(x.dtype)  # [B,T,1]
            summed = (x * mask).sum(dim=1)  # [B,D]
            denom = mask.sum(dim=1).clamp(min=1.0)  # [B,1]
            pooled = summed / denom
        else:
            raise ValueError(f"Unknown pooling={self.pooling}")

        # 3) 输出到所有 item 的 logits（分类）
        logits = self.out(pooled)  # [B, V]

        # 可选：不让模型预测 padding id=0（避免 topk 出现 0）
        logits[:, self.pad_id] = -1e9

        return ModelOutput(logits=logits)

