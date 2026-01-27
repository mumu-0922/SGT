# -*- coding: utf-8 -*-
"""
生成一个 JSONL toy 数据集，方便你练习“从文件读取数据 → DataLoader → 训练”。

这个 toy 数据集仍然遵循“可学习”的规律：下一物品 = 最后一个物品 + 1（循环）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="输出 jsonl 路径")
    parser.add_argument("--n", type=int, default=20000, help="样本数")
    parser.add_argument("--num-users", type=int, default=2000)
    parser.add_argument("--num-items", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            user_id = rng.randint(1, args.num_users)
            seq_len = rng.randint(6, args.max_len + 1)
            start = rng.randint(1, args.num_items)

            seq = []
            cur = start
            for _ in range(seq_len):
                seq.append(cur)
                cur = (cur % args.num_items) + 1

            obj = {
                "user_id": user_id,
                "history": seq[:-1],
                "target": seq[-1],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"wrote {args.n} samples to: {out_path}")


if __name__ == "__main__":
    main()

