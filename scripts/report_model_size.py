#!/usr/bin/env python3
"""
Report DecoderLM parameter count, parameter memory, and KV-cache sizing.

Example:
  python scripts/report_model_size.py \
    --n-vocab 4096 --n-embd 128 --n-head 8 --n-positions 512 \
    --seq-len 256 --batch-size 1 --num-blocks 64 --block-size 16 \
    --output-json results/model_size.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import minitorch
from minitorch.transformer import DecoderLM


def tensor_numel(value) -> int:
    size = getattr(value, "size", None)
    if size is not None:
        return int(size)
    shape = getattr(value, "shape", ())
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def collect_parameter_rows(model: DecoderLM):
    rows = []
    for name, param in model.named_parameters():
        n_params = tensor_numel(param.value)
        rows.append(
            {
                "name": name,
                "shape": list(param.value.shape),
                "parameters": n_params,
                "fp32_mb": n_params * 4 / (1024.0 * 1024.0),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Report DecoderLM model size and KV cache size.")
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-positions", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--p-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--output-json", default="results/model_size.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")

    backend = minitorch.TensorBackend()
    model = DecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        p_dropout=args.p_dropout,
        backend=backend,
        use_flash_attn=False,
    )
    param_rows = collect_parameter_rows(model)
    total_params = sum(row["parameters"] for row in param_rows)
    head_dim = args.n_embd // args.n_head
    dense_kv_bytes = (
        args.n_layers
        * args.batch_size
        * args.seq_len
        * args.n_head
        * head_dim
        * 2
        * 4
    )
    paged_capacity_bytes = (
        args.n_layers
        * args.num_blocks
        * args.block_size
        * args.n_head
        * head_dim
        * 2
        * 4
    )
    row: Dict[str, object] = {
        "n_vocab": args.n_vocab,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "head_dim": head_dim,
        "n_positions": args.n_positions,
        "n_layers": args.n_layers,
        "parameter_count": total_params,
        "parameter_fp32_mb": total_params * 4 / (1024.0 * 1024.0),
        "dense_kv_cache_mb_for_seq_len": dense_kv_bytes / (1024.0 * 1024.0),
        "paged_kv_capacity_mb": paged_capacity_bytes / (1024.0 * 1024.0),
        "paged_capacity_tokens": args.num_blocks * args.block_size,
        "parameters_by_tensor": param_rows,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print("Model Size Report")
    print("=================")
    print(f"parameters: {total_params:,}")
    print(f"parameter memory fp32: {row['parameter_fp32_mb']:.2f} MB")
    print(
        f"dense KV cache at batch={args.batch_size}, seq_len={args.seq_len}: "
        f"{row['dense_kv_cache_mb_for_seq_len']:.2f} MB"
    )
    print(
        f"paged KV capacity for num_blocks={args.num_blocks}, block_size={args.block_size}: "
        f"{row['paged_kv_capacity_mb']:.2f} MB "
        f"({row['paged_capacity_tokens']} token slots)"
    )
    print(f"Saved json to {args.output_json}")


if __name__ == "__main__":
    main()
