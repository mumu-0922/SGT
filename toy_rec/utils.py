# -*- coding: utf-8 -*-
"""
toy_rec.utils

一些“能复用”的小工具：
- set_seed：让实验可复现
- save_checkpoint/load_checkpoint：断点保存与恢复
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def set_seed(seed: int) -> None:
    """
    让实验尽量可复现（同样的数据/配置/seed 结果接近）。

    注意：深度学习完全一致复现并不总能保证（cuDNN 算子等），
    但 Step1 阶段你先把 seed 习惯养起来就很重要。
    """

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    extra: Dict[str, Any],
) -> None:
    """
    保存断点（最常见的训练必备功能）。
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra,
    }
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    加载断点。返回 extra（比如 epoch/global_step/args）。
    """

    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"])

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    return payload.get("extra", {})


def format_dict(d: Dict[str, Any]) -> str:
    """
    让日志更好读。
    """

    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)

