# -*- coding: utf-8 -*-
"""
toy_rec.metrics

这里实现最常用的 Top-K 离线评估指标（推荐/检索常见）：
- HitRate@K（HR@K）：目标是否出现在 Top-K
- NDCG@K：目标在 Top-K 的排名越靠前分越高

注意：
这里是“单一正样本”的 next-item 评估（每个样本只有一个 target）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class MetricResult:
    hr: float
    ndcg: float


def _rank_of_target(topk: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    topk: [B, K] 预测的 item id
    target: [B] 真实 target item id

    返回：rank（从 1 开始），没命中返回 0
    """

    # [B, K] -> [B, K] bool
    hit = topk.eq(target.view(-1, 1))
    # 找到第一个命中的位置（如果有）
    # argmax 对全 False 会返回 0，所以我们需要配合 any() 处理
    idx = hit.float().argmax(dim=1)  # [B]
    has_hit = hit.any(dim=1)  # [B]
    rank = torch.where(has_hit, idx + 1, torch.zeros_like(idx))
    return rank


def hr_ndcg_at_k(logits: torch.Tensor, target: torch.Tensor, k: int = 10) -> MetricResult:
    """
    logits: [B, V]
    target: [B]
    """

    k = int(k)
    topk = logits.topk(k, dim=1).indices  # [B, K]
    rank = _rank_of_target(topk, target)  # [B]

    hr = (rank > 0).float().mean().item()

    # NDCG：命中时 1/log2(rank+1)，否则 0
    denom = torch.log2(rank.float() + 1.0)
    ndcg = torch.where(rank > 0, 1.0 / denom, torch.zeros_like(denom)).mean().item()

    return MetricResult(hr=hr, ndcg=ndcg)

