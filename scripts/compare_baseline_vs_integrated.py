#!/usr/bin/env python3
"""
Compare baseline generation against the integrated flash+paged path.

Baseline:
  - Standard attention (`use_flash_attn=False`)
  - No paged KV cache
  - Recomputes full context each generation step

Integrated:
  - FlashAttention enabled for prefill (`use_flash_attn=True`)
  - PagedAttention KV cache for decode

Outputs:
  - A text report with latency, speedup, memory, and parity numbers
  - A CSV with the same per-config metrics
  - A small PNG latency plot when matplotlib is available

Methodology:
  - Speed metrics use warmed-run medians.
  - Memory metrics use cold runs on fresh model instances to avoid allocator reuse
    collapsing deltas to zero.
  - Correctness checks include token agreement and step-logit max absolute error.

Example:
  python scripts/compare_baseline_vs_integrated.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import minitorch
from minitorch.paged_attention import BlockManager
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.transformer import DecoderLM

try:
    import torch
except Exception:  # pragma: no cover - torch is expected on the GPU box
    torch = None

try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    _nvml_handle = None
    PYNVML_AVAILABLE = False


DEFAULT_PROMPT_LENGTHS = [64, 128, 256]
DEFAULT_BATCH_SIZE = 1
DEFAULT_NEW_TOKENS = 16
DEFAULT_REPEATS = 5
DEFAULT_MEMORY_REPEATS = 3
DEFAULT_N_VOCAB = 512
DEFAULT_N_EMBD = 256
DEFAULT_N_HEAD = 8
DEFAULT_N_POSITIONS = 2048
DEFAULT_BLOCK_SIZE = 16

_mem_baseline_mb = float("nan")
_mem_peak_mb = float("nan")


def get_backend():
    return minitorch.TensorBackend(minitorch.CudaKernelOps)


def sync_cuda() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def current_gpu_mem_mb() -> float:
    if PYNVML_AVAILABLE:
        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / (1024.0 * 1024.0)
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    return float("nan")


def reset_peak_memory() -> None:
    global _mem_baseline_mb, _mem_peak_mb
    _mem_baseline_mb = current_gpu_mem_mb()
    _mem_peak_mb = _mem_baseline_mb
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sample_peak_memory() -> None:
    global _mem_peak_mb
    cur = current_gpu_mem_mb()
    if np.isfinite(cur) and (not np.isfinite(_mem_peak_mb) or cur > _mem_peak_mb):
        _mem_peak_mb = cur


def peak_memory_mb() -> float:
    if PYNVML_AVAILABLE and np.isfinite(_mem_peak_mb) and np.isfinite(_mem_baseline_mb):
        return max(_mem_peak_mb - _mem_baseline_mb, 0.0)
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return float("nan")


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


def make_models(args, backend):
    base = DecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=False,
    )
    integrated = DecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=True,
    )
    copy_model_weights(base, integrated, backend)
    base.eval()
    integrated.eval()
    return base, integrated


def make_prompts(batch_size: int, prompt_len: int, n_vocab: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + prompt_len)
    return rng.integers(0, n_vocab, size=(batch_size, prompt_len), dtype=np.int32)


def _append_token_column(context: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    return np.concatenate([context, token_ids[:, None].astype(np.int32)], axis=1)


def _median(field: str, rows: Sequence[Dict[str, object]]) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def _max_step_logit_diff(
    baseline_steps: Sequence[np.ndarray],
    integrated_steps: Sequence[np.ndarray],
) -> float:
    if len(baseline_steps) != len(integrated_steps):
        raise ValueError("Mismatched number of decode steps when comparing logits")
    max_diff = 0.0
    for base_step, int_step in zip(baseline_steps, integrated_steps):
        if base_step.shape != int_step.shape:
            raise ValueError(
                f"Mismatched logit step shapes: {base_step.shape} vs {int_step.shape}"
            )
        step_diff = float(np.max(np.abs(base_step - int_step)))
        max_diff = max(max_diff, step_diff)
    return max_diff


def _measure_process_peak_gpu_mem_mb(proc: subprocess.Popen) -> float:
    if not PYNVML_AVAILABLE:
        return float("nan")

    peak_mb = 0.0
    while True:
        try:
            proc_infos = pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle)
        except Exception:
            proc_infos = []

        for info in proc_infos:
            if int(getattr(info, "pid", -1)) == int(proc.pid):
                used_bytes = float(getattr(info, "usedGpuMemory", 0))
                if used_bytes > 0:
                    peak_mb = max(peak_mb, used_bytes / (1024.0 * 1024.0))

        if proc.poll() is not None:
            break
        time.sleep(0.01)

    return peak_mb


def _read_process_rss_mb(pid: int) -> float:
    proc_status = f"/proc/{pid}/status"
    if os.path.exists(proc_status):
        try:
            with open(proc_status, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
        ).strip()
        if out:
            return float(out) / 1024.0
    except Exception:
        pass
    return float("nan")


def _measure_child_peaks(proc: subprocess.Popen) -> Dict[str, float]:
    peak_rss_mb = float("nan")
    peak_gpu_mb = 0.0 if PYNVML_AVAILABLE else float("nan")

    while True:
        rss_mb = _read_process_rss_mb(proc.pid)
        if np.isfinite(rss_mb):
            if not np.isfinite(peak_rss_mb):
                peak_rss_mb = rss_mb
            else:
                peak_rss_mb = max(peak_rss_mb, rss_mb)

        if PYNVML_AVAILABLE:
            gpu_mb = _measure_process_peak_gpu_mem_mb(proc)
            if np.isfinite(gpu_mb):
                peak_gpu_mb = max(peak_gpu_mb, gpu_mb)

        if proc.poll() is not None:
            break
        time.sleep(0.01)

    return {
        "peak_rss_mb": peak_rss_mb,
        "peak_gpu_mb": peak_gpu_mb,
    }


def _measure_mode_peak_gpu_mem_mb(args, prompt_len: int, mode: str) -> float:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--memory-worker",
        "--worker-mode",
        mode,
        "--worker-prompt-len",
        str(prompt_len),
        "--batch-size",
        str(args.batch_size),
        "--new-tokens",
        str(args.new_tokens),
        "--seed",
        str(args.seed),
        "--n-vocab",
        str(args.n_vocab),
        "--n-embd",
        str(args.n_embd),
        "--n-head",
        str(args.n_head),
        "--n-positions",
        str(args.n_positions),
        "--block-size",
        str(args.block_size),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    peaks = _measure_child_peaks(proc)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Memory worker failed for mode={mode}, prompt_len={prompt_len}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return peaks


def _memory_saved_pct(baseline_peak_mem: float, integrated_peak_mem: float) -> float:
    if not np.isfinite(baseline_peak_mem) or baseline_peak_mem <= 1e-9:
        return float("nan")
    return 100.0 * (baseline_peak_mem - integrated_peak_mem) / baseline_peak_mem


def _memory_ratio(baseline_peak_mem: float, integrated_peak_mem: float) -> float:
    if (
        not np.isfinite(baseline_peak_mem)
        or not np.isfinite(integrated_peak_mem)
        or integrated_peak_mem <= 1e-9
    ):
        return float("nan")
    return baseline_peak_mem / integrated_peak_mem


def _fmt_float(value: float, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f}{suffix}"


def run_baseline_generation(
    model: DecoderLM,
    prompts_np: np.ndarray,
    new_tokens: int,
    backend,
) -> Dict[str, object]:
    reset_peak_memory()
    contexts = prompts_np.copy()
    generated: List[List[int]] = [[] for _ in range(prompts_np.shape[0])]
    step_last_logits: List[np.ndarray] = []

    sync_cuda()
    sample_peak_memory()
    t0 = time.perf_counter()
    logits = model(tensor_from_numpy(contexts, backend=backend))
    sync_cuda()
    sample_peak_memory()
    t1 = time.perf_counter()

    first_logits = np.ascontiguousarray(logits.to_numpy()[:, -1, :], dtype=np.float32)
    step_last_logits.append(first_logits.copy())
    next_tokens = first_logits.argmax(axis=-1).astype(np.int32)
    for b, tok in enumerate(next_tokens):
        generated[b].append(int(tok))
    contexts = _append_token_column(contexts, next_tokens)

    sync_cuda()
    t2 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = model(tensor_from_numpy(contexts, backend=backend))
        sample_peak_memory()
        last_logits = np.ascontiguousarray(logits.to_numpy()[:, -1, :], dtype=np.float32)
        step_last_logits.append(last_logits.copy())
        next_tokens = last_logits.argmax(axis=-1).astype(np.int32)
        for b, tok in enumerate(next_tokens):
            generated[b].append(int(tok))
        contexts = _append_token_column(contexts, next_tokens)
    sync_cuda()
    sample_peak_memory()
    t3 = time.perf_counter()

    decode_tokens = prompts_np.shape[0] * (new_tokens - 1)
    decode_ms = (t3 - t2) * 1000.0
    total_ms = ((t1 - t0) + (t3 - t2)) * 1000.0
    return {
        "generated": generated,
        "step_last_logits": step_last_logits,
        "prefill_ms": (t1 - t0) * 1000.0,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "peak_mem_mb": peak_memory_mb(),
        "decode_tok_s": decode_tokens / (decode_ms / 1000.0),
    }


def run_integrated_generation(
    model: DecoderLM,
    prompts_np: np.ndarray,
    new_tokens: int,
    backend,
    block_size: int,
    n_head: int,
    n_embd: int,
) -> Dict[str, object]:
    batch_size, prompt_len = prompts_np.shape
    reset_peak_memory()
    num_blocks = batch_size * math.ceil((prompt_len + new_tokens + 1) / block_size) + 4
    bm = BlockManager(
        num_layers=4,
        num_blocks=num_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=n_embd // n_head,
        backend=backend,
    )
    sample_peak_memory()
    seq_ids = list(range(batch_size))
    generated: List[List[int]] = [[] for _ in range(batch_size)]
    step_last_logits: List[np.ndarray] = []

    sync_cuda()
    sample_peak_memory()
    t0 = time.perf_counter()
    first_tokens = []
    first_logits = []
    for sid in seq_ids:
        prompt_t = tensor_from_numpy(prompts_np[sid : sid + 1], backend=backend)
        logits = model.prefill(prompt_t, sid, bm)
        sample_peak_memory()
        last_logits = np.ascontiguousarray(logits.to_numpy()[0, -1, :], dtype=np.float32)
        first_logits.append(last_logits)
        tok = int(last_logits.argmax())
        generated[sid].append(tok)
        first_tokens.append(tok)
    step_last_logits.append(np.stack(first_logits, axis=0))
    sync_cuda()
    sample_peak_memory()
    t1 = time.perf_counter()

    token_ids_np = np.array(first_tokens, dtype=np.int32)[:, None]
    sync_cuda()
    t2 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = model.decode_step_batch(
            tensor_from_numpy(token_ids_np, backend=backend),
            seq_ids,
            bm,
        )
        sample_peak_memory()
        last_logits = np.ascontiguousarray(logits.to_numpy()[:, 0, :], dtype=np.float32)
        step_last_logits.append(last_logits.copy())
        next_tokens = last_logits.argmax(axis=-1).astype(np.int32)
        for sid, tok in enumerate(next_tokens):
            generated[sid].append(int(tok))
        token_ids_np = next_tokens[:, None]
    sync_cuda()
    sample_peak_memory()
    t3 = time.perf_counter()

    decode_tokens = batch_size * (new_tokens - 1)
    decode_ms = (t3 - t2) * 1000.0
    total_ms = ((t1 - t0) + (t3 - t2)) * 1000.0
    return {
        "generated": generated,
        "step_last_logits": step_last_logits,
        "prefill_ms": (t1 - t0) * 1000.0,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "peak_mem_mb": peak_memory_mb(),
        "decode_tok_s": decode_tokens / (decode_ms / 1000.0),
    }


def benchmark_config(args, backend, prompt_len: int) -> Dict[str, object]:
    print(
        f"[compare] prompt_len={prompt_len} batch_size={args.batch_size} "
        f"new_tokens={args.new_tokens} repeats={args.repeats} "
        f"memory_repeats={args.memory_repeats}",
        flush=True,
    )
    prompts_np = make_prompts(args.batch_size, prompt_len, args.n_vocab, args.seed)

    baseline_mem_runs = []
    integrated_mem_runs = []
    for rep in range(args.memory_repeats):
        print(f"[compare]   memory run {rep + 1}/{args.memory_repeats}", flush=True)
        baseline_mem_runs.append(
            _measure_mode_peak_gpu_mem_mb(args, prompt_len, "baseline")
        )
        integrated_mem_runs.append(
            _measure_mode_peak_gpu_mem_mb(args, prompt_len, "integrated")
        )

    base_model, int_model = make_models(args, backend)
    warm_base = run_baseline_generation(base_model, prompts_np, min(2, args.new_tokens), backend)
    warm_int = run_integrated_generation(
        int_model,
        prompts_np,
        min(2, args.new_tokens),
        backend,
        args.block_size,
        args.n_head,
        args.n_embd,
    )
    del warm_base, warm_int

    baseline_runs = []
    integrated_runs = []
    repeat_correctness = []
    repeat_logit_diffs = []
    for rep in range(args.repeats):
        print(f"[compare]   speed repeat {rep + 1}/{args.repeats}: baseline", flush=True)
        base_run = run_baseline_generation(base_model, prompts_np, args.new_tokens, backend)
        print(f"[compare]   speed repeat {rep + 1}/{args.repeats}: integrated", flush=True)
        int_run = run_integrated_generation(
            int_model,
            prompts_np,
            args.new_tokens,
            backend,
            args.block_size,
            args.n_head,
            args.n_embd,
        )
        baseline_runs.append(base_run)
        integrated_runs.append(int_run)
        repeat_correctness.append(base_run["generated"] == int_run["generated"])
        repeat_logit_diffs.append(
            _max_step_logit_diff(
                base_run["step_last_logits"],
                int_run["step_last_logits"],
            )
        )

    baseline_peak_rss = _median("peak_rss_mb", baseline_mem_runs)
    integrated_peak_rss = _median("peak_rss_mb", integrated_mem_runs)
    baseline_peak_gpu = _median("peak_gpu_mb", baseline_mem_runs)
    integrated_peak_gpu = _median("peak_gpu_mb", integrated_mem_runs)
    correctness = all(repeat_correctness)

    return {
        "prompt_len": prompt_len,
        "batch_size": args.batch_size,
        "new_tokens": args.new_tokens,
        "correct": correctness,
        "correct_runs": int(sum(repeat_correctness)),
        "max_logit_diff": float(np.max(repeat_logit_diffs)),
        "median_logit_diff": float(np.median(repeat_logit_diffs)),
        "baseline_prefill_ms": _median("prefill_ms", baseline_runs),
        "baseline_decode_ms": _median("decode_ms", baseline_runs),
        "baseline_total_ms": _median("total_ms", baseline_runs),
        "baseline_peak_rss_mb": baseline_peak_rss,
        "baseline_peak_gpu_mb": baseline_peak_gpu,
        "baseline_decode_tok_s": _median("decode_tok_s", baseline_runs),
        "integrated_prefill_ms": _median("prefill_ms", integrated_runs),
        "integrated_decode_ms": _median("decode_ms", integrated_runs),
        "integrated_total_ms": _median("total_ms", integrated_runs),
        "integrated_peak_rss_mb": integrated_peak_rss,
        "integrated_peak_gpu_mb": integrated_peak_gpu,
        "integrated_decode_tok_s": _median("decode_tok_s", integrated_runs),
        "prefill_speedup": _median("prefill_ms", baseline_runs)
        / max(_median("prefill_ms", integrated_runs), 1e-9),
        "decode_speedup": _median("decode_ms", baseline_runs)
        / max(_median("decode_ms", integrated_runs), 1e-9),
        "speedup_total": _median("total_ms", baseline_runs)
        / max(_median("total_ms", integrated_runs), 1e-9),
        "rss_saved_mb": baseline_peak_rss - integrated_peak_rss,
        "rss_saved_pct": _memory_saved_pct(baseline_peak_rss, integrated_peak_rss),
        "rss_ratio": _memory_ratio(baseline_peak_rss, integrated_peak_rss),
        "gpu_saved_mb": baseline_peak_gpu - integrated_peak_gpu,
        "gpu_saved_pct": _memory_saved_pct(baseline_peak_gpu, integrated_peak_gpu),
        "gpu_ratio": _memory_ratio(baseline_peak_gpu, integrated_peak_gpu),
    }


def write_report(results: List[Dict[str, object]], args) -> str:
    lines = []
    lines.append("Baseline vs Integrated Attention Comparison")
    lines.append("=" * 48)
    lines.append(f"batch_size={args.batch_size}")
    lines.append(f"prompt_lengths={args.prompt_lengths}")
    lines.append(f"new_tokens={args.new_tokens}")
    lines.append(f"repeats={args.repeats}")
    lines.append(f"memory_repeats={args.memory_repeats}")
    lines.append(
        f"n_vocab={args.n_vocab}, n_embd={args.n_embd}, n_head={args.n_head}, n_positions={args.n_positions}"
    )
    lines.append(f"block_size={args.block_size}")
    lines.append(f"memory_source={'child_process_rss' + ('+pynvml_gpu' if PYNVML_AVAILABLE else '')}")
    lines.append("")
    lines.append("Baseline: standard attention, no paged KV cache")
    lines.append("Integrated: FlashAttention prefill + PagedAttention decode")
    lines.append("Speed values are warmed-run medians across repeats.")
    lines.append("Memory values are isolated child-process peak RSS medians; this includes NumPy/Python allocations and fits the MiniTorch implementation better than device-only deltas.")
    if PYNVML_AVAILABLE:
        lines.append("GPU-only child-process peaks are also recorded in the CSV for reference.")
    lines.append("Correctness requires generated-token agreement on every repeated run.")
    lines.append("Numeric parity reports max absolute difference on step logits.")
    if args.batch_size != 1:
        lines.append(
            "Warning: total/prefill speedup is only apples-to-apples when batch_size=1 because integrated prefill is per-sequence."
        )
    lines.append("")
    lines.append(
        f"{'Prompt':>8} {'Correct':>8} {'Base Tot':>10} {'Int Tot':>10} "
        f"{'Tot Spd':>8} {'Dec Spd':>8} {'Base RSS':>10} {'Int RSS':>10} "
        f"{'RSS Save':>10} {'dLogit':>10} {'Int Tok/s':>10}"
    )
    lines.append("-" * 124)

    for row in results:
        lines.append(
            f"{row['prompt_len']:>8} {str(row['correct']):>8} "
            f"{row['baseline_total_ms']:>10.2f} {row['integrated_total_ms']:>10.2f} "
            f"{row['speedup_total']:>7.2f}x {row['decode_speedup']:>7.2f}x "
            f"{row['baseline_peak_rss_mb']:>10.1f} {row['integrated_peak_rss_mb']:>10.1f} "
            f"{_fmt_float(float(row['rss_saved_pct']), '%'):>10} "
            f"{row['max_logit_diff']:>10.3e} "
            f"{row['integrated_decode_tok_s']:>10.1f}"
        )

    overall = (
        float(np.sum([row["baseline_total_ms"] for row in results]))
        / max(float(np.sum([row["integrated_total_ms"] for row in results])), 1e-9)
    )
    overall_decode = (
        float(np.sum([row["baseline_decode_ms"] for row in results]))
        / max(float(np.sum([row["integrated_decode_ms"] for row in results])), 1e-9)
    )
    mean_total_speedup = float(np.mean([row["speedup_total"] for row in results]))
    mean_decode_speedup = float(np.mean([row["decode_speedup"] for row in results]))
    overall_rss_saved_mb = float(np.sum([row["rss_saved_mb"] for row in results]))
    mean_rss_saved_mb = float(np.mean([row["rss_saved_mb"] for row in results]))
    mean_rss_saved_pct = float(np.mean([row["rss_saved_pct"] for row in results]))
    mean_rss_ratio = float(np.mean([row["rss_ratio"] for row in results]))
    worst_logit_diff = float(np.max([row["max_logit_diff"] for row in results]))
    all_correct = all(bool(row["correct"]) for row in results)
    lines.append("")
    lines.append(f"Overall total-latency speedup (sum of medians): {overall:.2f}x")
    lines.append(f"Overall decode-latency speedup (sum of medians): {overall_decode:.2f}x")
    lines.append(f"Mean per-config total-latency speedup: {mean_total_speedup:.2f}x")
    lines.append(f"Mean per-config decode-latency speedup: {mean_decode_speedup:.2f}x")
    lines.append(
        f"Mean per-config RSS saved: {mean_rss_saved_mb:.1f} MB ({_fmt_float(mean_rss_saved_pct, '%')})"
    )
    lines.append(
        f"Mean per-config baseline/integrated RSS ratio: "
        f"{(_fmt_float(mean_rss_ratio, 'x') if np.isfinite(mean_rss_ratio) else 'n/a')}"
    )
    lines.append(f"Summed RSS saved across listed configs: {_fmt_float(overall_rss_saved_mb, ' MB')}")
    lines.append(f"Worst step-logit max abs diff: {worst_logit_diff:.3e}")
    lines.append(f"Generated tokens matched across modes: {all_correct}")
    if PYNVML_AVAILABLE:
        mean_gpu_saved_mb = float(np.mean([row["gpu_saved_mb"] for row in results]))
        mean_gpu_saved_pct = float(np.mean([row["gpu_saved_pct"] for row in results]))
        lines.append(
            f"Mean per-config GPU-only saved memory: {mean_gpu_saved_mb:.1f} MB ({_fmt_float(mean_gpu_saved_pct, '%')})"
        )
    lines.append("")
    lines.append("Detailed per-config metrics:")
    for row in results:
        lines.append(
            f"prompt={row['prompt_len']}: "
            f"baseline[prefill={row['baseline_prefill_ms']:.2f} ms, decode={row['baseline_decode_ms']:.2f} ms, tok/s={row['baseline_decode_tok_s']:.1f}, rss={row['baseline_peak_rss_mb']:.1f} MB, gpu={_fmt_float(float(row['baseline_peak_gpu_mb']), ' MB')}] "
            f"integrated[prefill={row['integrated_prefill_ms']:.2f} ms, decode={row['integrated_decode_ms']:.2f} ms, tok/s={row['integrated_decode_tok_s']:.1f}, rss={row['integrated_peak_rss_mb']:.1f} MB, gpu={_fmt_float(float(row['integrated_peak_gpu_mb']), ' MB')}] "
            f"rss_saved={row['rss_saved_mb']:.1f} MB ({_fmt_float(float(row['rss_saved_pct']), '%')}, {_fmt_float(float(row['rss_ratio']), 'x')}) "
            f"logit_diff[max={row['max_logit_diff']:.3e}, median={row['median_logit_diff']:.3e}] "
            f"correctness={row['correct_runs']}/{args.repeats} repeats"
        )
    return "\n".join(lines) + "\n"


def write_csv(results: List[Dict[str, object]], output_path: str) -> str:
    fieldnames = [
        "prompt_len",
        "batch_size",
        "new_tokens",
        "correct",
        "correct_runs",
        "max_logit_diff",
        "median_logit_diff",
        "baseline_prefill_ms",
        "baseline_decode_ms",
        "baseline_total_ms",
        "baseline_peak_rss_mb",
        "baseline_peak_gpu_mb",
        "baseline_decode_tok_s",
        "integrated_prefill_ms",
        "integrated_decode_ms",
        "integrated_total_ms",
        "integrated_peak_rss_mb",
        "integrated_peak_gpu_mb",
        "integrated_decode_tok_s",
        "prefill_speedup",
        "decode_speedup",
        "speedup_total",
        "rss_saved_mb",
        "rss_saved_pct",
        "rss_ratio",
        "gpu_saved_mb",
        "gpu_saved_pct",
        "gpu_ratio",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})
    return output_path


def save_plot(results: List[Dict[str, object]], output_path: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    prompt_lengths = [int(row["prompt_len"]) for row in results]
    baseline_total = [float(row["baseline_total_ms"]) for row in results]
    integrated_total = [float(row["integrated_total_ms"]) for row in results]

    plt.figure(figsize=(5.2, 3.2))
    plt.plot(prompt_lengths, baseline_total, marker="o", label="Baseline")
    plt.plot(prompt_lengths, integrated_total, marker="o", label="Integrated")
    plt.xlabel("Prompt Length")
    plt.ylabel("Total Generation Latency (ms)")
    plt.title("Baseline vs Integrated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def validate_args(args) -> None:
    if args.new_tokens < 2:
        raise ValueError("--new-tokens must be >= 2 so decode metrics are non-zero")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.memory_repeats < 1:
        raise ValueError("--memory-repeats must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if any(p <= 0 for p in args.prompt_lengths):
        raise ValueError("--prompt-lengths must all be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    if max(args.prompt_lengths) + args.new_tokens + 1 > args.n_positions:
        raise ValueError(
            "--n-positions must be at least max(prompt_lengths) + new_tokens + 1"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline generation against integrated flash+paged generation."
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=DEFAULT_PROMPT_LENGTHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--memory-repeats", type=int, default=DEFAULT_MEMORY_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-vocab", type=int, default=DEFAULT_N_VOCAB)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-positions", type=int, default=DEFAULT_N_POSITIONS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--output-txt", default="baseline_vs_integrated.txt")
    parser.add_argument("--output-csv", default="baseline_vs_integrated.csv")
    parser.add_argument("--output-plot", default="baseline_vs_integrated.png")
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=["baseline", "integrated"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-len", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    validate_args(args)
    return args


def run_memory_worker(args) -> None:
    if args.worker_mode is None or args.worker_prompt_len is None:
        raise ValueError("Memory worker requires --worker-mode and --worker-prompt-len")

    backend = get_backend()
    prompts_np = make_prompts(args.batch_size, args.worker_prompt_len, args.n_vocab, args.seed)
    base_model, int_model = make_models(args, backend)

    if args.worker_mode == "baseline":
        _ = run_baseline_generation(base_model, prompts_np, args.new_tokens, backend)
    else:
        _ = run_integrated_generation(
            int_model,
            prompts_np,
            args.new_tokens,
            backend,
            args.block_size,
            args.n_head,
            args.n_embd,
        )

    sync_cuda()
    time.sleep(0.05)


def main():
    args = parse_args()
    if args.memory_worker:
        run_memory_worker(args)
        return

    os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_plot) or ".", exist_ok=True)

    backend = get_backend()
    print("[compare] starting benchmark", flush=True)
    results = [benchmark_config(args, backend, prompt_len) for prompt_len in args.prompt_lengths]

    report = write_report(results, args)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(report)
    csv_path = write_csv(results, args.output_csv)
    print(report, end="")

    plot_path = save_plot(results, args.output_plot)
    if plot_path is not None:
        print(f"Saved plot to {plot_path}")
    else:
        print("matplotlib not available; skipped plot.")
    print(f"Saved report to {args.output_txt}")
    print(f"Saved csv to {csv_path}")


if __name__ == "__main__":
    main()
