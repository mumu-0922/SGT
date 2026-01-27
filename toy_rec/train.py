# -*- coding: utf-8 -*-
"""
toy_rec.train

这是 Step1 的核心：一个“可复用”的训练脚本。

你应该重点看懂：
1) 参数/配置如何控制实验
2) DataLoader -> batch dict -> model -> logits -> loss 的数据形状
3) 训练 loop：forward / loss / backward / step / zero_grad
4) 保存/加载 checkpoint（复现与排错都离不开）
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from toy_rec.dataset import JsonlSeqRecDataset, ToySeqRecDataset, collate_seqrec
from toy_rec.metrics import hr_ndcg_at_k
from toy_rec.model import SimpleSeqRecModel
from toy_rec.utils import format_dict, load_checkpoint, save_checkpoint, set_seed


def build_dataloaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """
    构建 train/val 两个 dataloader。

    注意：collate_fn 里把 batch 移到 GPU（device），这样训练 loop 更清爽。
    """

    if args.dataset == "toy":
        ds = ToySeqRecDataset(num_items=args.num_items, max_len=args.max_len, n=args.n_samples, seed=args.seed)
    elif args.dataset == "jsonl":
        if not args.train_path:
            raise ValueError("--train-path is required when --dataset jsonl")
        ds = JsonlSeqRecDataset(args.train_path)
    else:
        raise ValueError(f"Unknown dataset={args.dataset}")

    # 简单切一个验证集（10%）
    val_size = max(1, int(len(ds) * 0.1))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    def _collate(batch):
        return collate_seqrec(batch, max_len=args.max_len, pad_id=0, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, k: int) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_hr = 0.0
    total_ndcg = 0.0
    n = 0

    for batch in loader:
        out = model(batch)
        logits = out.logits  # [B, V]
        labels = batch["labels"]  # [B]

        loss = loss_fn(logits, labels)
        m = hr_ndcg_at_k(logits, labels, k=k)

        bsz = labels.size(0)
        total_loss += loss.item() * bsz
        total_hr += m.hr * bsz
        total_ndcg += m.ndcg * bsz
        n += bsz

    return {
        "val_loss": total_loss / max(1, n),
        f"val_hr@{k}": total_hr / max(1, n),
        f"val_ndcg@{k}": total_ndcg / max(1, n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    # 数据相关
    parser.add_argument("--dataset", type=str, choices=["toy", "jsonl"], default="toy")
    parser.add_argument("--train-path", type=str, default="", help="JSONL 数据路径（dataset=jsonl 时需要）")
    parser.add_argument("--n-samples", type=int, default=20000, help="toy 数据集样本数（dataset=toy 时有效）")
    parser.add_argument("--num-items", type=int, default=500, help="物品数（item id 范围：1..num_items）")
    parser.add_argument("--max-len", type=int, default=50, help="最大历史长度（会截断/padding 到该长度）")

    # 模型相关
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--pooling", type=str, choices=["last", "mean"], default="last")

    # 训练相关
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数（显存不够时常用）")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="梯度裁剪（稳定训练常用）")
    parser.add_argument("--amp", action="store_true", help="开启混合精度（省显存/更快，需 CUDA）")

    # 评估与保存
    parser.add_argument("--eval-every", type=int, default=200, help="每多少 step 评估一次")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="toy_rec/runs", help="保存目录")
    parser.add_argument("--resume", type=str, default="", help="从 checkpoint 恢复训练（可选）")

    # 其它
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[info] device={device} torch={torch.__version__}")

    train_loader, val_loader = build_dataloaders(args, device=device)

    model = SimpleSeqRecModel(num_items=args.num_items, embed_dim=args.embed_dim, pooling=args.pooling).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    # 断点恢复（可选）
    start_epoch = 0
    global_step = 0
    if args.resume:
        extra = load_checkpoint(args.resume, model=model, optimizer=optimizer, scaler=scaler, map_location=device)
        start_epoch = int(extra.get("epoch", 0))
        global_step = int(extra.get("global_step", 0))
        print(f"[info] resumed from {args.resume} (epoch={start_epoch}, global_step={global_step})")

    run_dir = Path(args.save_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"

    print(f"[info] run_dir={run_dir}")
    print(f"[info] train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # 训练 loop
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            model.train()

            # 混合精度：autocast 只在 CUDA+amp 开启时生效
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(batch)
                logits = out.logits
                labels = batch["labels"]
                loss = loss_fn(logits, labels)

                # 梯度累积：把 loss 除以 accum，保证等效 batch 不变
                loss = loss / max(1, args.grad_accum)

            scaler.scale(loss).backward()

            # 到了累积步数再更新一次参数
            if (global_step + 1) % args.grad_accum == 0:
                # 梯度裁剪要在 unscale 之后做
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 记录训练日志（把真实 loss 还原回来）
            if global_step % 50 == 0:
                cur = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.item() * max(1, args.grad_accum)),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                print("[train]", format_dict(cur))

            # 定期评估 + 保存
            if (global_step + 1) % args.eval_every == 0:
                metrics = evaluate(model, val_loader, loss_fn, k=args.topk)
                print("[eval ]", format_dict({"epoch": epoch, "step": global_step, **metrics}))

                save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler if scaler.is_enabled() else None,
                    extra={
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": vars(args),
                    },
                )
                print(f"[info] saved checkpoint to: {ckpt_path}")

            global_step += 1

        # 每个 epoch 结束也评估一次
        metrics = evaluate(model, val_loader, loss_fn, k=args.topk)
        print("[eval ]", format_dict({"epoch": epoch, "step": global_step, **metrics}))

    print("[done] training finished")
    print(f"[done] last checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

