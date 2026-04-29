#!/usr/bin/env python3
"""
Train dense-attention and FlashAttention DecoderLM models on the same batches.

This script is for final-report loss evidence.  PagedAttention is intentionally
excluded because the current implementation is an inference/decode KV-cache path,
not a training path.

Example:
  python scripts/compare_training_loss_flash_vs_dense.py \
    --epochs 3 \
    --samples-per-epoch 1024 \
    --validation-samples 256 \
    --batch-size 16 \
    --model-max-length 40 \
    --n-vocab 4096 \
    --n-embd 128 \
    --output-csv results/training_loss_flash_vs_dense.csv \
    --output-plot results/training_loss_flash_vs_dense.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import minitorch
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.transformer import DecoderLM
from project.run_machine_translation import (
    clip_grad_norm,
    collate_batch,
    get_dataset,
    get_tokenizer,
    loss_fn,
)


def get_backend(use_cpu: bool):
    if use_cpu:
        return minitorch.TensorBackend()
    from minitorch.cuda_kernel_ops import CudaKernelOps

    return minitorch.TensorBackend(CudaKernelOps)


def clone_tensor(value, backend):
    arr = np.ascontiguousarray(value.to_numpy(), dtype=np.float32)
    out = tensor_from_numpy(arr, backend=backend)
    out.requires_grad_(True)
    return out


def copy_model_weights(src: DecoderLM, dst: DecoderLM, backend) -> None:
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    for name, src_param in src_params.items():
        if name not in dst_params:
            raise KeyError(f"Missing parameter {name} in destination model")
        dst_params[name].update(clone_tensor(src_param.value, backend))


def make_model(args, backend, use_flash_attn: bool) -> DecoderLM:
    return DecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.model_max_length,
        p_dropout=args.p_dropout,
        backend=backend,
        use_flash_attn=use_flash_attn,
    )


def iter_fixed_batches(
    examples: Sequence[dict],
    n_samples: int,
    batch_size: int,
    seed: int,
) -> Iterable[List[dict]]:
    rng = np.random.default_rng(seed)
    count = min(n_samples, len(examples))
    indices = rng.permutation(len(examples))[:count]
    for start in range(0, count, batch_size):
        yield [examples[int(i)] for i in indices[start : start + batch_size]]


def train_one_epoch(model, optimizer, batches, collate_fn, grad_clip: float) -> Dict[str, float]:
    model.train()
    losses = []
    tokens = 0
    t0 = time.perf_counter()

    for examples in batches:
        batch = collate_fn(examples=examples)
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            continue
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss_value)
        tokens += int(np.prod(batch["input_ids"].shape))

    elapsed = time.perf_counter() - t0
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "tokens_per_sec": tokens / elapsed if elapsed > 0 else float("nan"),
        "steps": len(losses),
    }


def evaluate(model, examples, batch_size: int, collate_fn) -> float:
    model.eval()
    losses = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = collate_fn(examples=batch_examples)
        losses.append(float(loss_fn(batch=batch, model=model).item()))
    return float(np.mean(losses)) if losses else float("nan")


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "epoch",
        "mode",
        "use_flash_attn",
        "train_loss",
        "validation_loss",
        "tokens_per_sec",
        "steps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_plot(rows: List[Dict[str, object]], path: str) -> None:
    if not path:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping training-loss plot")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    modes = ["dense", "flash"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        xs = [int(row["epoch"]) for row in mode_rows]
        axes[0].plot(xs, [float(row["train_loss"]) for row in mode_rows], marker="o", label=mode)
        axes[1].plot(xs, [float(row["validation_loss"]) for row in mode_rows], marker="o", label=mode)
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare training loss for dense vs FlashAttention.")
    parser.add_argument("--dataset-name", default="bbaaaa/iwslt14-de-en-preprocess")
    parser.add_argument("--model-max-length", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--n-vocab", type=int, default=4096)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--p-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--workdir", default="workdir_training_loss_compare")
    parser.add_argument("--output-csv", default="results/training_loss_flash_vs_dense.csv")
    parser.add_argument("--output-plot", default="results/training_loss_flash_vs_dense.png")
    parser.add_argument("--config-json", default="results/training_loss_flash_vs_dense_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.config_json) or ".", exist_ok=True)

    backend = get_backend(args.use_cpu)
    dataset, src_key, tgt_key = get_dataset(args.dataset_name, args.model_max_length)
    validation_examples = dataset["validation"][: args.validation_samples]
    tokenizer = get_tokenizer(
        examples=dataset["train"],
        vocab_size=args.n_vocab,
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=args.workdir,
    )
    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        backend=backend,
    )

    dense_model = make_model(args, backend, use_flash_attn=False)
    flash_model = make_model(args, backend, use_flash_attn=True)
    copy_model_weights(dense_model, flash_model, backend)
    dense_optim = minitorch.Adam(dense_model.parameters(), lr=args.learning_rate)
    flash_optim = minitorch.Adam(flash_model.parameters(), lr=args.learning_rate)

    rows: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_batches = list(
            iter_fixed_batches(
                dataset["train"],
                args.samples_per_epoch,
                args.batch_size,
                seed=args.seed + epoch,
            )
        )
        for mode, model, optimizer, use_flash in [
            ("dense", dense_model, dense_optim, False),
            ("flash", flash_model, flash_optim, True),
        ]:
            train_metrics = train_one_epoch(
                model,
                optimizer,
                epoch_batches,
                collate_fn,
                grad_clip=args.grad_clip,
            )
            val_loss = evaluate(model, validation_examples, args.batch_size, collate_fn)
            row = {
                "epoch": epoch,
                "mode": mode,
                "use_flash_attn": use_flash,
                "train_loss": train_metrics["loss"],
                "validation_loss": val_loss,
                "tokens_per_sec": train_metrics["tokens_per_sec"],
                "steps": train_metrics["steps"],
            }
            rows.append(row)
            print(
                f"epoch={epoch} mode={mode} train_loss={row['train_loss']:.4f} "
                f"validation_loss={row['validation_loss']:.4f} "
                f"tokens_per_sec={row['tokens_per_sec']:.1f}"
            )

    write_csv(rows, args.output_csv)
    write_plot(rows, args.output_plot)
    with open(args.config_json, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Saved csv to {args.output_csv}")
    if args.output_plot:
        print(f"Saved plot to {args.output_plot}")
    print(f"Saved config to {args.config_json}")


if __name__ == "__main__":
    main()
