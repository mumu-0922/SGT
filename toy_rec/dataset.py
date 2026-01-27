# -*- coding: utf-8 -*-
"""
toy_rec.dataset

核心目标（给 Step1 用）：
1) 让你弄懂 `Dataset` / `DataLoader` / `collate_fn` 是怎么把“变长序列”组装成 batch 的。
2) 让后续复用时，只要换掉 Dataset（读取真实数据），训练 loop 基本不用改。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SeqRecSample:
    """
    一个最常见的“序列推荐/下一物品预测”样本：
    - history: 用户历史交互序列（变长）
    - target: 下一物品（label）

    约定：
    - item_id 从 1..num_items
    - 0 作为 padding（填充）专用 id
    """

    history: List[int]
    target: int
    user_id: Optional[int] = None  # 可选：后续你想加 user embedding 时会用到


class ToySeqRecDataset(Dataset[SeqRecSample]):
    """
    一个“可学习”的 toy 数据集（用于验证训练链路是否正确）。

    生成规则（保证模型能学到）：
    - 先随机选一个起始 item
    - 序列按 “+1 循环递增” 生成：i, i+1, i+2, ...
    - history = 前 L 个，target = 第 L+1 个（下一物品）

    这样只要模型学会“看最后一个 item”，就能很快把 loss 降下来。
    """

    def __init__(self, num_items: int, max_len: int, n: int, seed: int = 42) -> None:
        self.num_items = int(num_items)
        self.max_len = int(max_len)
        self.n = int(n)
        self.seed = int(seed)

    def __len__(self) -> int:  # noqa: D401
        return self.n

    def __getitem__(self, idx: int) -> SeqRecSample:
        # 为了可复现：每个 idx 对应一个确定的随机序列
        rng = random.Random(self.seed + int(idx))

        # 真实训练里 history 长度经常变化；这里模拟“变长”
        # 注意：target 需要在 history 后面，所以这里 seq_len >= 2
        seq_len = rng.randint(6, self.max_len + 1)
        start = rng.randint(1, self.num_items)

        seq: List[int] = []
        cur = start
        for _ in range(seq_len):
            seq.append(cur)
            cur = (cur % self.num_items) + 1  # +1 循环

        history = seq[:-1]
        target = seq[-1]
        return SeqRecSample(history=history, target=target, user_id=None)


class JsonlSeqRecDataset(Dataset[SeqRecSample]):
    """
    从 JSONL 文件读取样本，方便你把 toy_rec 的训练骨架迁移到真实项目。

    JSONL 每行示例：
    {"user_id": 1, "history": [12, 98, 3], "target": 4}
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._samples: List[SeqRecSample] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                history = list(map(int, obj["history"]))
                target = int(obj["target"])
                user_id = obj.get("user_id")
                user_id = int(user_id) if user_id is not None else None
                self._samples.append(SeqRecSample(history=history, target=target, user_id=user_id))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SeqRecSample:
        return self._samples[int(idx)]


def _truncate_history(history: Sequence[int], max_len: int) -> List[int]:
    """
    真实场景里用户历史可能很长，常见做法是只保留“最近 max_len 个”。
    """

    if len(history) <= max_len:
        return list(history)
    return list(history[-max_len:])


def collate_seqrec(
    batch: Sequence[SeqRecSample],
    *,
    max_len: int,
    pad_id: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    这是 Step1 最关键的函数：collate_fn。

    DataLoader 每次会给你一个 list[Sample]（每条样本 history 长度不一样），
    collate_fn 的工作就是把它们“拼成一个 batch 张量”。

    输出（通用约定，后面复用很方便）：
    - input_ids: [B, T]，padding 后的 item id 序列
    - attention_mask: [B, T]，1 表示有效 token，0 表示 padding
    - labels: [B]，下一物品 id（分类标签）
    """

    bsz = len(batch)
    max_len = int(max_len)

    histories = [_truncate_history(x.history, max_len) for x in batch]
    lengths = torch.tensor([len(h) for h in histories], dtype=torch.long)

    # 1) padding：把不同长度的 history 填充到同一长度 T
    #    这里固定 T=max_len（也可以用 batch 内最大长度，但固定更方便对齐后续模型）
    input_ids = torch.full((bsz, max_len), fill_value=int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, h in enumerate(histories):
        if not h:
            continue
        t = min(len(h), max_len)
        input_ids[i, :t] = torch.tensor(h[:t], dtype=torch.long)
        attention_mask[i, :t] = 1

    labels = torch.tensor([int(x.target) for x in batch], dtype=torch.long)

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,  # 可选：有些 pooling/模型会用到长度
    }


def iter_jsonl(path: str | Path) -> Iterable[Dict]:
    """
    小工具：按行读取 jsonl。
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

