#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitorch.flash_attention2 import flash_attention2_backward
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.tensor_ops import TensorBackend
from minitorch.cuda_kernel_ops import CudaKernelOps


CONFIGS = {
    "bs4_seq512_embd512_heads8": {"batch_size": 4, "seq_len": 512, "embd": 512, "heads": 8},
    "bs8_seq256_embd256_heads8": {"batch_size": 8, "seq_len": 256, "embd": 256, "heads": 8},
    "bs8_seq256_embd512_heads8": {"batch_size": 8, "seq_len": 256, "embd": 512, "heads": 8},
    "bs8_seq64_embd256_heads8": {"batch_size": 8, "seq_len": 64, "embd": 256, "heads": 8},
    "bs2_seq1024_embd512_heads8": {"batch_size": 2, "seq_len": 1024, "embd": 512, "heads": 8},
    "bs1_seq2048_embd512_heads8": {"batch_size": 1, "seq_len": 2048, "embd": 512, "heads": 8},
}

VALIDATION_CASES = [
    (1, 1, 8, 8),
    (1, 2, 16, 8),
    (1, 2, 16, 16),
    (1, 4, 32, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPU naive attention backward vs FA2 CUDA backward.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()),
        help="Named benchmark configurations to run.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["naive", "fa2"],
        choices=["naive", "fa2"],
        help="Variants to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per worker.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed iterations per worker.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save aggregated JSON results.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=["naive", "fa2"], help=argparse.SUPPRESS)
    parser.add_argument("--config-name", choices=list(CONFIGS.keys()), help=argparse.SUPPRESS)
    return parser.parse_args()


def compute_out_lse(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    bsz, nhead, seqlen, _ = q.shape
    out = np.empty_like(q, dtype=np.float32)
    lse = np.empty((bsz, nhead, seqlen), dtype=np.float32)
    for batch_idx in range(bsz):
        for head_idx in range(nhead):
            scores = (q[batch_idx, head_idx] @ k[batch_idx, head_idx].T) * scale
            max_scores = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            z = np.sum(exp_scores, axis=-1, keepdims=True)
            probs = exp_scores / z
            out[batch_idx, head_idx] = probs @ v[batch_idx, head_idx]
            lse[batch_idx, head_idx] = (np.log(z) + max_scores)[:, 0]
    return out, lse


def dense_backward_numpy(
    dout: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    out: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz, nhead, seqlen, _ = q.shape
    dq = np.zeros_like(q, dtype=np.float32)
    dk = np.zeros_like(k, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    for batch_idx in range(bsz):
        for head_idx in range(nhead):
            scores = (q[batch_idx, head_idx] @ k[batch_idx, head_idx].T) * scale
            max_scores = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            dV = probs.T @ dout[batch_idx, head_idx]
            dP = dout[batch_idx, head_idx] @ v[batch_idx, head_idx].T
            d_row = np.sum(dout[batch_idx, head_idx] * out[batch_idx, head_idx], axis=-1, keepdims=True)
            dS = probs * (dP - d_row)
            dq[batch_idx, head_idx] = (dS @ k[batch_idx, head_idx]) * scale
            dk[batch_idx, head_idx] = (dS.T @ q[batch_idx, head_idx]) * scale
            dv[batch_idx, head_idx] = dV
    return dq, dk, dv


def build_inputs(config_name: str, seed: int) -> dict[str, np.ndarray | float | int | str]:
    cfg = CONFIGS[config_name]
    batch_size = cfg["batch_size"]
    seq_len = cfg["seq_len"]
    embd = cfg["embd"]
    heads = cfg["heads"]
    head_dim = embd // heads
    scale = 1.0 / np.sqrt(float(head_dim))
    rng = np.random.default_rng(seed)
    shape = (batch_size, heads, seq_len, head_dim)
    q = rng.normal(size=shape).astype(np.float32)
    k = rng.normal(size=shape).astype(np.float32)
    v = rng.normal(size=shape).astype(np.float32)
    dout = rng.normal(size=shape).astype(np.float32)
    out, lse = compute_out_lse(q, k, v, scale)
    return {
        "config_name": config_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embd": embd,
        "heads": heads,
        "head_dim": head_dim,
        "scale": float(scale),
        "q": q,
        "k": k,
        "v": v,
        "dout": dout,
        "out": out,
        "lse": lse,
    }


def build_validation_inputs(
    batch_size: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    scale = 1.0 / np.sqrt(float(head_dim))
    rng = np.random.default_rng(seed)
    shape = (batch_size, heads, seq_len, head_dim)
    q = rng.normal(size=shape).astype(np.float32)
    k = rng.normal(size=shape).astype(np.float32)
    v = rng.normal(size=shape).astype(np.float32)
    dout = rng.normal(size=shape).astype(np.float32)
    out, lse = compute_out_lse(q, k, v, scale)
    return q, k, v, dout, out, lse, float(scale)


def run_naive_worker(payload: dict[str, np.ndarray | float | int | str]) -> dict[str, float]:
    backend = TensorBackend(CudaKernelOps)
    scale = float(payload["scale"])
    dq, dk, dv = CudaKernelOps.dense_attention_backward(
        dout=tensor_from_numpy(payload["dout"], backend=backend),
        q=tensor_from_numpy(payload["q"], backend=backend),
        k=tensor_from_numpy(payload["k"], backend=backend),
        v=tensor_from_numpy(payload["v"], backend=backend),
        out=tensor_from_numpy(payload["out"], backend=backend),
        logsumexp=tensor_from_numpy(payload["lse"], backend=backend),
        causal=False,
        softmax_scale=scale,
    )

    checksum = float(np.mean(dq.to_numpy()) + np.mean(dk.to_numpy()) + np.mean(dv.to_numpy()))
    return {"checksum": checksum}


def validate_dense_cuda_wrapper() -> None:
    backend = TensorBackend(CudaKernelOps)
    for case_index, (batch_size, heads, seq_len, head_dim) in enumerate(VALIDATION_CASES):
        q, k, v, dout, out, lse, scale = build_validation_inputs(
            batch_size=batch_size,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            seed=123 + case_index,
        )
        dq_ref, dk_ref, dv_ref = dense_backward_numpy(dout, q, k, v, out, scale)
        dq, dk, dv = CudaKernelOps.dense_attention_backward(
            dout=tensor_from_numpy(dout, backend=backend),
            q=tensor_from_numpy(q, backend=backend),
            k=tensor_from_numpy(k, backend=backend),
            v=tensor_from_numpy(v, backend=backend),
            out=tensor_from_numpy(out, backend=backend),
            logsumexp=tensor_from_numpy(lse, backend=backend),
            causal=False,
            softmax_scale=scale,
        )
        dq_np = dq.to_numpy()
        dk_np = dk.to_numpy()
        dv_np = dv.to_numpy()
        if not (
            np.allclose(dq_np, dq_ref, atol=1e-4, rtol=1e-4)
            and np.allclose(dk_np, dk_ref, atol=1e-4, rtol=1e-4)
            and np.allclose(dv_np, dv_ref, atol=1e-4, rtol=1e-4)
        ):
            raise RuntimeError(
                "Dense CUDA wrapper validation failed against NumPy reference "
                f"for case {(batch_size, heads, seq_len, head_dim)}"
            )


def validate_fa2_cuda_wrapper() -> None:
    backend = TensorBackend(CudaKernelOps)
    for case_index, (batch_size, heads, seq_len, head_dim) in enumerate(VALIDATION_CASES):
        q, k, v, dout, out, lse, scale = build_validation_inputs(
            batch_size=batch_size,
            heads=heads,
            seq_len=seq_len,
            head_dim=head_dim,
            seed=321 + case_index,
        )
        dq_ref, dk_ref, dv_ref = dense_backward_numpy(dout, q, k, v, out, scale)
        dq, dk, dv = flash_attention2_backward(
            dout=tensor_from_numpy(dout, backend=backend),
            q=tensor_from_numpy(q, backend=backend),
            k=tensor_from_numpy(k, backend=backend),
            v=tensor_from_numpy(v, backend=backend),
            out=tensor_from_numpy(out, backend=backend),
            logsumexp=tensor_from_numpy(lse, backend=backend),
            causal=False,
            softmax_scale=scale,
            use_cuda_kernel=True,
        )
        dq_np = dq.to_numpy()
        dk_np = dk.to_numpy()
        dv_np = dv.to_numpy()
        if not (
            np.allclose(dq_np, dq_ref, atol=1e-4, rtol=1e-4)
            and np.allclose(dk_np, dk_ref, atol=1e-4, rtol=1e-4)
            and np.allclose(dv_np, dv_ref, atol=1e-4, rtol=1e-4)
        ):
            raise RuntimeError(
                "FA2 CUDA wrapper validation failed against NumPy reference "
                f"for case {(batch_size, heads, seq_len, head_dim)}"
            )


def run_fa2_worker(payload: dict[str, np.ndarray | float | int | str]) -> dict[str, float]:
    backend = TensorBackend(CudaKernelOps)
    scale = float(payload["scale"])

    dq, dk, dv = flash_attention2_backward(
        dout=tensor_from_numpy(payload["dout"], backend=backend),
        q=tensor_from_numpy(payload["q"], backend=backend),
        k=tensor_from_numpy(payload["k"], backend=backend),
        v=tensor_from_numpy(payload["v"], backend=backend),
        out=tensor_from_numpy(payload["out"], backend=backend),
        logsumexp=tensor_from_numpy(payload["lse"], backend=backend),
        causal=False,
        softmax_scale=scale,
        use_cuda_kernel=True,
    )
    checksum = float(np.mean(dq.to_numpy()) + np.mean(dk.to_numpy()) + np.mean(dv.to_numpy()))
    return {"checksum": checksum}


def run_worker(variant: str, config_name: str, warmup: int, repeats: int, seed: int) -> dict[str, object]:
    payload = build_inputs(config_name, seed)
    backend = TensorBackend(CudaKernelOps)
    scale = float(payload["scale"])
    q = tensor_from_numpy(payload["q"], backend=backend)
    k = tensor_from_numpy(payload["k"], backend=backend)
    v = tensor_from_numpy(payload["v"], backend=backend)
    dout = tensor_from_numpy(payload["dout"], backend=backend)
    out = tensor_from_numpy(payload["out"], backend=backend)
    lse = tensor_from_numpy(payload["lse"], backend=backend)

    checksum_worker = run_fa2_worker if variant == "fa2" else run_naive_worker
    checksum = float(checksum_worker(payload)["checksum"])

    iteration_ms: list[float] = []
    allocated_bytes_values: list[int] = []
    outer_trials = 3
    for _ in range(outer_trials):
        if variant == "fa2":
            avg_ms, allocated_bytes = CudaKernelOps.benchmark_flash_attention2_backward(
                dout=dout,
                q=q,
                k=k,
                v=v,
                out=out,
                logsumexp=lse,
                causal=False,
                softmax_scale=scale,
                warmup=warmup,
                repeats=repeats,
            )
        else:
            avg_ms, allocated_bytes = CudaKernelOps.benchmark_dense_attention_backward(
                dout=dout,
                q=q,
                k=k,
                v=v,
                out=out,
                logsumexp=lse,
                causal=False,
                softmax_scale=scale,
                warmup=warmup,
                repeats=repeats,
            )
        iteration_ms.append(float(avg_ms))
        allocated_bytes_values.append(int(allocated_bytes))

    avg_ms = float(np.mean(iteration_ms))
    tokens_per_second = (int(payload["batch_size"]) * int(payload["seq_len"]) * 1000.0 / avg_ms) if avg_ms > 0 else 0.0
    allocated_bytes = int(max(allocated_bytes_values)) if allocated_bytes_values else 0

    return {
        "config_name": config_name,
        "variant": variant,
        "batch_size": int(payload["batch_size"]),
        "seq_len": int(payload["seq_len"]),
        "embd": int(payload["embd"]),
        "heads": int(payload["heads"]),
        "head_dim": int(payload["head_dim"]),
        "warmup": warmup,
        "repeats": repeats,
        "iteration_ms": iteration_ms,
        "avg_ms": avg_ms,
        "min_ms": float(np.min(iteration_ms)),
        "max_ms": float(np.max(iteration_ms)),
        "checksum": checksum,
        "throughput_tokens_per_sec": tokens_per_second,
        "allocated_bytes": allocated_bytes,
        "allocated_mib": allocated_bytes / (1024.0 * 1024.0),
    }
def run_subprocess_worker(script_path: Path, variant: str, config_name: str, warmup: int, repeats: int, seed: int) -> dict[str, object]:
    import subprocess

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--variant",
        variant,
        "--config-name",
        config_name,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed for variant={variant} config={config_name}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Worker produced no output for variant={variant} config={config_name}")
    result = json.loads(lines[-1])
    result["worker_stdout"] = stdout
    result["worker_stderr"] = stderr
    return result


def aggregate_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["config_name"]), {})[str(result["variant"])] = result

    summary: list[dict[str, object]] = []
    for config_name in sorted(grouped.keys()):
        variants = grouped[config_name]
        item: dict[str, object] = {
            "config_name": config_name,
            "shape": {
                "batch_size": variants[next(iter(variants))]["batch_size"],
                "seq_len": variants[next(iter(variants))]["seq_len"],
                "embd": variants[next(iter(variants))]["embd"],
                "heads": variants[next(iter(variants))]["heads"],
                "head_dim": variants[next(iter(variants))]["head_dim"],
            },
            "variants": variants,
        }
        if "naive" in variants and "fa2" in variants:
            naive_ms = float(variants["naive"]["avg_ms"])
            fa2_ms = float(variants["fa2"]["avg_ms"])
            item["fa2_speedup_vs_naive"] = naive_ms / fa2_ms if fa2_ms > 0 else None
            naive_mem = variants["naive"].get("allocated_bytes")
            fa2_mem = variants["fa2"].get("allocated_bytes")
            if isinstance(naive_mem, int) and isinstance(fa2_mem, int) and fa2_mem > 0:
                item["naive_vs_fa2_alloc_ratio"] = naive_mem / fa2_mem
        summary.append(item)
    return summary


def print_human_summary(summary: list[dict[str, object]]) -> None:
    print("=== GPU Naive vs FA2 Benchmark Summary ===")
    for item in summary:
        shape = item["shape"]
        print(
            f"CONFIG {item['config_name']}: "
            f"B={shape['batch_size']} T={shape['seq_len']} E={shape['embd']} H={shape['heads']} D={shape['head_dim']}"
        )
        variants = item["variants"]
        for variant_name in ("naive", "fa2"):
            if variant_name not in variants:
                continue
            result = variants[variant_name]
            print(
                f"  {variant_name:>5} avg_ms={float(result['avg_ms']):8.3f} "
                f"min_ms={float(result['min_ms']):8.3f} max_ms={float(result['max_ms']):8.3f} "
                f"throughput_tok_s={float(result['throughput_tokens_per_sec']):10.1f} "
                f"alloc_mib={float(result['allocated_mib']):8.1f}"
            )
        if "fa2_speedup_vs_naive" in item:
            print(f"   speedup naive/fa2 = {float(item['fa2_speedup_vs_naive']):.3f}x")
        if "naive_vs_fa2_alloc_ratio" in item:
            print(f"   alloc_bytes naive/fa2 = {float(item['naive_vs_fa2_alloc_ratio']):.3f}x")


def main() -> None:
    args = parse_args()

    if args.worker:
        result = run_worker(
            variant=str(args.variant),
            config_name=str(args.config_name),
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            seed=int(args.seed),
        )
        print(json.dumps(result))
        return

    validate_dense_cuda_wrapper()
    validate_fa2_cuda_wrapper()

    script_path = Path(__file__).resolve()
    all_results: list[dict[str, object]] = []

    for config_index, config_name in enumerate(args.configs):
        for variant in args.variants:
            derived_seed = int(args.seed) + config_index * 100
            result = run_subprocess_worker(
                script_path=script_path,
                variant=variant,
                config_name=config_name,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                seed=derived_seed,
            )
            all_results.append(result)

    summary = aggregate_results(all_results)
    print_human_summary(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results": all_results,
            "summary": summary,
            "configs": args.configs,
            "variants": args.variants,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "hostname": os.uname().nodename,
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON results to {args.output_json}")


if __name__ == "__main__":
    main()