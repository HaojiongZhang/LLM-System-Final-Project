#!/usr/bin/env python3
"""
Benchmark aligned attention/inference modes in the current repo.

Tracked modes:
  - dense_recompute: standard attention, no KV cache
  - flash_recompute: flash-enabled attention, no KV cache
  - dense_paged: standard attention for prefill, paged KV decode
  - flash_paged: flash-enabled prefill, paged KV decode

This script is meant to answer two separate questions with aligned settings:
  1. What changes when flash is enabled without paging?
  2. What changes when paging is enabled, with and without flash prefill?

Outputs:
  - text report
  - CSV with per-mode metrics
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import minitorch
from minitorch.paged_attention import BlockManager
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.transformer import DecoderLM

try:
    import torch
except Exception:  # pragma: no cover
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


MODE_NAMES = [
    "dense_recompute",
    "flash_recompute",
    "dense_paged",
    "flash_paged",
]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    use_flash_attn: bool
    use_paged: bool


MODE_SPECS = [
    ModeSpec("dense_recompute", use_flash_attn=False, use_paged=False),
    ModeSpec("flash_recompute", use_flash_attn=True, use_paged=False),
    ModeSpec("dense_paged", use_flash_attn=False, use_paged=True),
    ModeSpec("flash_paged", use_flash_attn=True, use_paged=True),
]


_mem_baseline_mb = float("nan")
_mem_peak_mb = float("nan")
_rss_baseline_mb = float("nan")
_rss_peak_mb = float("nan")


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


def current_rss_mb() -> float:
    return _read_process_rss_mb(os.getpid())


def reset_peak_memory() -> None:
    global _mem_baseline_mb, _mem_peak_mb, _rss_baseline_mb, _rss_peak_mb
    _mem_baseline_mb = current_gpu_mem_mb()
    _mem_peak_mb = _mem_baseline_mb
    _rss_baseline_mb = current_rss_mb()
    _rss_peak_mb = _rss_baseline_mb
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sample_peak_memory() -> None:
    global _mem_peak_mb, _rss_peak_mb
    cur = current_gpu_mem_mb()
    if np.isfinite(cur) and (not np.isfinite(_mem_peak_mb) or cur > _mem_peak_mb):
        _mem_peak_mb = cur
    rss = current_rss_mb()
    if np.isfinite(rss) and (not np.isfinite(_rss_peak_mb) or rss > _rss_peak_mb):
        _rss_peak_mb = rss


def peak_memory_mb() -> float:
    if PYNVML_AVAILABLE and np.isfinite(_mem_peak_mb) and np.isfinite(_mem_baseline_mb):
        return max(_mem_peak_mb - _mem_baseline_mb, 0.0)
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return float("nan")


def peak_rss_delta_mb() -> float:
    if np.isfinite(_rss_peak_mb) and np.isfinite(_rss_baseline_mb):
        return max(_rss_peak_mb - _rss_baseline_mb, 0.0)
    return float("nan")


def _memory_ratio(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(candidate) or candidate <= 1e-9:
        return float("nan")
    return reference / candidate


def _memory_savings_pct(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or reference <= 1e-9 or not np.isfinite(candidate):
        return float("nan")
    return 100.0 * (reference - candidate) / reference


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
            try:
                proc_infos = pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle)
            except Exception:
                proc_infos = []
            for info in proc_infos:
                if int(getattr(info, "pid", -1)) == int(proc.pid):
                    used_bytes = float(getattr(info, "usedGpuMemory", 0))
                    if used_bytes > 0:
                        peak_gpu_mb = max(peak_gpu_mb, used_bytes / (1024.0 * 1024.0))

        if proc.poll() is not None:
            break
        time.sleep(0.01)

    return {
        "peak_rss_mb": peak_rss_mb,
        "peak_gpu_mb": peak_gpu_mb,
    }


def _measure_mode_peaks(args, prompt_len: int, mode_name: str) -> Dict[str, float]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--memory-worker",
        "--worker-mode",
        mode_name,
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
            f"Memory worker failed for mode={mode_name}, prompt_len={prompt_len}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return peaks


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
    model = DecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=use_flash_attn,
    )
    model.eval()
    return model


def make_prompts(batch_size: int, prompt_len: int, n_vocab: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + prompt_len)
    return rng.integers(0, n_vocab, size=(batch_size, prompt_len), dtype=np.int32)


def _append_token_column(context: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    return np.concatenate([context, token_ids[:, None].astype(np.int32)], axis=1)


def _median(rows: Sequence[Dict[str, object]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def _max_step_logit_diff(reference_steps: Sequence[np.ndarray], candidate_steps: Sequence[np.ndarray]) -> float:
    if len(reference_steps) != len(candidate_steps):
        raise ValueError("Mismatched step counts")
    max_diff = 0.0
    for ref_step, cand_step in zip(reference_steps, candidate_steps):
        max_diff = max(max_diff, float(np.max(np.abs(ref_step - cand_step))))
    return max_diff


def run_recompute_generation(
    model: DecoderLM,
    prompts_np: np.ndarray,
    new_tokens: int,
    backend,
) -> Dict[str, object]:
    contexts = prompts_np.copy()
    generated: List[List[int]] = [[] for _ in range(prompts_np.shape[0])]
    step_last_logits: List[np.ndarray] = []

    reset_peak_memory()
    sync_cuda()
    sample_peak_memory()
    t0 = time.perf_counter()
    logits = model(tensor_from_numpy(contexts, backend=backend))
    sync_cuda()
    sample_peak_memory()
    t1 = time.perf_counter()

    last_logits = np.ascontiguousarray(logits.to_numpy()[:, -1, :], dtype=np.float32)
    step_last_logits.append(last_logits.copy())
    next_tokens = last_logits.argmax(axis=-1).astype(np.int32)
    for b, tok in enumerate(next_tokens):
        generated[b].append(int(tok))
    contexts = _append_token_column(contexts, next_tokens)
    prefill_peak_gpu_mb = peak_memory_mb()
    prefill_peak_rss_delta_mb = peak_rss_delta_mb()

    reset_peak_memory()
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
    decode_peak_gpu_mb = peak_memory_mb()
    decode_peak_rss_delta_mb = peak_rss_delta_mb()

    decode_ms = (t3 - t2) * 1000.0
    decode_tokens = prompts_np.shape[0] * (new_tokens - 1)
    return {
        "generated": generated,
        "step_last_logits": step_last_logits,
        "prefill_ms": (t1 - t0) * 1000.0,
        "decode_ms": decode_ms,
        "total_ms": ((t1 - t0) + (t3 - t2)) * 1000.0,
        "peak_mem_mb": max(prefill_peak_gpu_mb, decode_peak_gpu_mb),
        "prefill_peak_gpu_mb": prefill_peak_gpu_mb,
        "decode_peak_gpu_mb": decode_peak_gpu_mb,
        "prefill_peak_rss_delta_mb": prefill_peak_rss_delta_mb,
        "decode_peak_rss_delta_mb": decode_peak_rss_delta_mb,
        "decode_tok_s": decode_tokens / (decode_ms / 1000.0),
    }


def run_paged_generation(
    model: DecoderLM,
    prompts_np: np.ndarray,
    new_tokens: int,
    backend,
    block_size: int,
    n_head: int,
    n_embd: int,
) -> Dict[str, object]:
    batch_size, prompt_len = prompts_np.shape
    num_blocks = batch_size * int(np.ceil((prompt_len + new_tokens + 1) / block_size)) + 4
    block_manager = BlockManager(
        num_layers=4,
        num_blocks=num_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=n_embd // n_head,
        backend=backend,
    )
    seq_ids = list(range(batch_size))
    generated: List[List[int]] = [[] for _ in range(batch_size)]
    step_last_logits: List[np.ndarray] = []

    reset_peak_memory()
    sync_cuda()
    sample_peak_memory()
    t0 = time.perf_counter()
    first_tokens = []
    first_logits = []
    for sid in seq_ids:
        prompt_t = tensor_from_numpy(prompts_np[sid : sid + 1], backend=backend)
        logits = model.prefill(prompt_t, sid, block_manager)
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
    prefill_peak_gpu_mb = peak_memory_mb()
    prefill_peak_rss_delta_mb = peak_rss_delta_mb()

    token_ids_np = np.array(first_tokens, dtype=np.int32)[:, None]
    reset_peak_memory()
    sync_cuda()
    t2 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = model.decode_step_batch(
            tensor_from_numpy(token_ids_np, backend=backend),
            seq_ids,
            block_manager,
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
    decode_peak_gpu_mb = peak_memory_mb()
    decode_peak_rss_delta_mb = peak_rss_delta_mb()

    decode_ms = (t3 - t2) * 1000.0
    decode_tokens = batch_size * (new_tokens - 1)
    return {
        "generated": generated,
        "step_last_logits": step_last_logits,
        "prefill_ms": (t1 - t0) * 1000.0,
        "decode_ms": decode_ms,
        "total_ms": ((t1 - t0) + (t3 - t2)) * 1000.0,
        "peak_mem_mb": max(prefill_peak_gpu_mb, decode_peak_gpu_mb),
        "prefill_peak_gpu_mb": prefill_peak_gpu_mb,
        "decode_peak_gpu_mb": decode_peak_gpu_mb,
        "prefill_peak_rss_delta_mb": prefill_peak_rss_delta_mb,
        "decode_peak_rss_delta_mb": decode_peak_rss_delta_mb,
        "decode_tok_s": decode_tokens / (decode_ms / 1000.0),
    }


def run_mode(mode: ModeSpec, model: DecoderLM, prompts_np: np.ndarray, args, backend) -> Dict[str, object]:
    if mode.use_paged:
        return run_paged_generation(
            model,
            prompts_np,
            args.new_tokens,
            backend,
            args.block_size,
            args.n_head,
            args.n_embd,
        )
    return run_recompute_generation(model, prompts_np, args.new_tokens, backend)


def benchmark_config(args, backend, prompt_len: int) -> List[Dict[str, object]]:
    print(
        f"[matrix] prompt_len={prompt_len} batch_size={args.batch_size} "
        f"new_tokens={args.new_tokens} repeats={args.repeats} memory_repeats={args.memory_repeats}",
        flush=True,
    )
    prompts_np = make_prompts(args.batch_size, prompt_len, args.n_vocab, args.seed)

    baseline_model = make_model(args, backend, use_flash_attn=False)
    flash_model = make_model(args, backend, use_flash_attn=True)
    copy_model_weights(baseline_model, flash_model, backend)

    # Warm the speed path on shared models.
    for mode in MODE_SPECS:
        model = flash_model if mode.use_flash_attn else baseline_model
        _ = run_mode(
            mode,
            model,
            prompts_np,
            argparse.Namespace(
                new_tokens=min(2, args.new_tokens),
                block_size=args.block_size,
                n_head=args.n_head,
                n_embd=args.n_embd,
            ),
            backend,
        )

    speed_runs: Dict[str, List[Dict[str, object]]] = {mode.name: [] for mode in MODE_SPECS}
    for rep in range(args.repeats):
        for mode in MODE_SPECS:
            print(f"[matrix]   speed repeat {rep + 1}/{args.repeats}: {mode.name}", flush=True)
            model = flash_model if mode.use_flash_attn else baseline_model
            speed_runs[mode.name].append(run_mode(mode, model, prompts_np, args, backend))

    memory_runs: Dict[str, List[Dict[str, object]]] = {mode.name: [] for mode in MODE_SPECS}
    for rep in range(args.memory_repeats):
        print(f"[matrix]   memory run {rep + 1}/{args.memory_repeats}", flush=True)
        for mode in MODE_SPECS:
            memory_runs[mode.name].append(_measure_mode_peaks(args, prompt_len, mode.name))

    reference_name = "dense_recompute"
    reference_speed = speed_runs[reference_name]
    rows: List[Dict[str, object]] = []
    for mode in MODE_SPECS:
        mode_speed = speed_runs[mode.name]
        parity_tokens = [
            ref_run["generated"] == cand_run["generated"]
            for ref_run, cand_run in zip(reference_speed, mode_speed)
        ]
        parity_logits = [
            _max_step_logit_diff(ref_run["step_last_logits"], cand_run["step_last_logits"])
            for ref_run, cand_run in zip(reference_speed, mode_speed)
        ]
        row = {
            "prompt_len": prompt_len,
            "mode": mode.name,
            "use_flash_attn": mode.use_flash_attn,
            "use_paged": mode.use_paged,
            "batch_size": args.batch_size,
            "new_tokens": args.new_tokens,
            "prefill_ms": _median(mode_speed, "prefill_ms"),
            "decode_ms": _median(mode_speed, "decode_ms"),
            "total_ms": _median(mode_speed, "total_ms"),
            "prefill_peak_gpu_mb": _median(mode_speed, "prefill_peak_gpu_mb"),
            "decode_peak_gpu_mb": _median(mode_speed, "decode_peak_gpu_mb"),
            "prefill_peak_rss_delta_mb": _median(mode_speed, "prefill_peak_rss_delta_mb"),
            "decode_peak_rss_delta_mb": _median(mode_speed, "decode_peak_rss_delta_mb"),
            "peak_rss_mb": _median(memory_runs[mode.name], "peak_rss_mb"),
            "peak_gpu_mb": _median(memory_runs[mode.name], "peak_gpu_mb"),
            "decode_tok_s": _median(mode_speed, "decode_tok_s"),
            "tokens_match_reference": all(parity_tokens),
            "token_match_runs": int(sum(parity_tokens)),
            "max_logit_diff_vs_dense_recompute": float(np.max(parity_logits)),
            "median_logit_diff_vs_dense_recompute": float(np.median(parity_logits)),
        }
        rows.append(row)
    return rows


def enrich_pairwise(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_prompt: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_prompt.setdefault(int(row["prompt_len"]), {})[str(row["mode"])] = row

    for prompt_rows in by_prompt.values():
        dense = prompt_rows["dense_recompute"]
        flash = prompt_rows["flash_recompute"]
        dense_paged = prompt_rows["dense_paged"]
        flash_paged = prompt_rows["flash_paged"]

        for row in prompt_rows.values():
            row["speedup_vs_dense_recompute"] = float(dense["total_ms"]) / max(float(row["total_ms"]), 1e-9)
            row["prefill_speedup_vs_dense_recompute"] = float(dense["prefill_ms"]) / max(float(row["prefill_ms"]), 1e-9)
            row["decode_speedup_vs_dense_recompute"] = float(dense["decode_ms"]) / max(float(row["decode_ms"]), 1e-9)
            row["rss_ratio_vs_dense_recompute"] = float(dense["peak_rss_mb"]) / max(float(row["peak_rss_mb"]), 1e-9)
            row["prefill_gpu_mem_ratio_vs_dense_recompute"] = _memory_ratio(
                float(dense["prefill_peak_gpu_mb"]), float(row["prefill_peak_gpu_mb"])
            )
            row["decode_gpu_mem_ratio_vs_dense_recompute"] = _memory_ratio(
                float(dense["decode_peak_gpu_mb"]), float(row["decode_peak_gpu_mb"])
            )
            row["prefill_gpu_mem_savings_pct_vs_dense_recompute"] = _memory_savings_pct(
                float(dense["prefill_peak_gpu_mb"]), float(row["prefill_peak_gpu_mb"])
            )
            row["decode_gpu_mem_savings_pct_vs_dense_recompute"] = _memory_savings_pct(
                float(dense["decode_peak_gpu_mb"]), float(row["decode_peak_gpu_mb"])
            )

        flash["flash_only_total_speedup_vs_dense"] = float(dense["total_ms"]) / max(float(flash["total_ms"]), 1e-9)
        dense_paged["paged_only_total_speedup_vs_dense"] = float(dense["total_ms"]) / max(float(dense_paged["total_ms"]), 1e-9)
        flash_paged["full_total_speedup_vs_dense"] = float(dense["total_ms"]) / max(float(flash_paged["total_ms"]), 1e-9)
        flash_paged["full_total_speedup_vs_flash_only"] = float(flash["total_ms"]) / max(float(flash_paged["total_ms"]), 1e-9)
        flash_paged["full_total_speedup_vs_paged_only"] = float(dense_paged["total_ms"]) / max(float(flash_paged["total_ms"]), 1e-9)

    return rows


def write_report(rows: List[Dict[str, object]], args) -> str:
    lines = []
    lines.append("Attention Mode Matrix Benchmark")
    lines.append("=" * 32)
    lines.append(f"prompt_lengths={args.prompt_lengths}")
    lines.append(f"batch_size={args.batch_size}")
    lines.append(f"new_tokens={args.new_tokens}")
    lines.append(f"repeats={args.repeats}")
    lines.append(f"memory_repeats={args.memory_repeats}")
    lines.append(
        f"n_vocab={args.n_vocab}, n_embd={args.n_embd}, n_head={args.n_head}, n_positions={args.n_positions}, block_size={args.block_size}"
    )
    lines.append("")
    lines.append("Reference mode: dense_recompute")
    lines.append("flash_recompute is the current branch's closest equivalent to fforward-style non-paged flash attention.")
    lines.append("flash_paged is the current full integrated path: flash-enabled prefill plus paged KV decode.")
    lines.append("Memory values include isolated child-process whole-run peaks plus in-process stage peak deltas sampled around prefill and decode.")
    lines.append("")
    lines.append(
        f"{'Prompt':>8} {'Mode':>16} {'Pre(ms)':>10} {'Dec(ms)':>10} {'Tok/s':>10} {'RSS(MB)':>10} {'TokEq':>8} {'PreSpd':>8} {'DecSpd':>8}"
    )
    lines.append("-" * 106)
    for row in rows:
        lines.append(
            f"{row['prompt_len']:>8} {row['mode']:>16} "
            f"{row['prefill_ms']:>10.2f} {row['decode_ms']:>10.2f} "
            f"{row['decode_tok_s']:>10.1f} {row['peak_rss_mb']:>10.1f} "
            f"{str(row['tokens_match_reference']):>8} "
            f"{row['prefill_speedup_vs_dense_recompute']:>7.2f}x "
            f"{row['decode_speedup_vs_dense_recompute']:>7.2f}x"
        )

    lines.append("")
    lines.append("Prompt-level summaries:")
    prompt_lengths = sorted({int(row["prompt_len"]) for row in rows})
    for prompt_len in prompt_lengths:
        prompt_rows = {str(row["mode"]): row for row in rows if int(row["prompt_len"]) == prompt_len}
        dense = prompt_rows["dense_recompute"]
        flash = prompt_rows["flash_recompute"]
        dense_paged = prompt_rows["dense_paged"]
        flash_paged = prompt_rows["flash_paged"]
        lines.append(
            f"prompt={prompt_len}: "
            f"flash_only={float(dense['total_ms']) / max(float(flash['total_ms']), 1e-9):.2f}x, "
            f"paged_only={float(dense['total_ms']) / max(float(dense_paged['total_ms']), 1e-9):.2f}x, "
            f"full={float(dense['total_ms']) / max(float(flash_paged['total_ms']), 1e-9):.2f}x, "
            f"integrated_prefill={float(dense['prefill_ms']) / max(float(flash_paged['prefill_ms']), 1e-9):.2f}x, "
            f"integrated_decode={float(dense['decode_ms']) / max(float(flash_paged['decode_ms']), 1e-9):.2f}x, "
            f"full_vs_flash_only={float(flash['total_ms']) / max(float(flash_paged['total_ms']), 1e-9):.2f}x, "
            f"full_vs_paged_only={float(dense_paged['total_ms']) / max(float(flash_paged['total_ms']), 1e-9):.2f}x"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: List[Dict[str, object]], output_path: str) -> str:
    fieldnames = [
        "prompt_len",
        "mode",
        "use_flash_attn",
        "use_paged",
        "batch_size",
        "new_tokens",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "prefill_peak_gpu_mb",
        "decode_peak_gpu_mb",
        "prefill_peak_rss_delta_mb",
        "decode_peak_rss_delta_mb",
        "peak_rss_mb",
        "peak_gpu_mb",
        "decode_tok_s",
        "tokens_match_reference",
        "token_match_runs",
        "max_logit_diff_vs_dense_recompute",
        "median_logit_diff_vs_dense_recompute",
        "speedup_vs_dense_recompute",
        "prefill_speedup_vs_dense_recompute",
        "decode_speedup_vs_dense_recompute",
        "rss_ratio_vs_dense_recompute",
        "prefill_gpu_mem_ratio_vs_dense_recompute",
        "decode_gpu_mem_ratio_vs_dense_recompute",
        "prefill_gpu_mem_savings_pct_vs_dense_recompute",
        "decode_gpu_mem_savings_pct_vs_dense_recompute",
        "flash_only_total_speedup_vs_dense",
        "paged_only_total_speedup_vs_dense",
        "full_total_speedup_vs_dense",
        "full_total_speedup_vs_flash_only",
        "full_total_speedup_vs_paged_only",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path


def validate_args(args) -> None:
    if args.new_tokens < 2:
        raise ValueError("--new-tokens must be >= 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.repeats < 1 or args.memory_repeats < 1:
        raise ValueError("--repeats and --memory-repeats must be >= 1")
    if any(prompt_len <= 0 for prompt_len in args.prompt_lengths):
        raise ValueError("--prompt-lengths must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    if max(args.prompt_lengths) + args.new_tokens + 1 > args.n_positions:
        raise ValueError("--n-positions too small for prompt + generation length")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark dense/flash/paged mode matrix.")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-positions", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output-txt", default="attention_mode_matrix.txt")
    parser.add_argument("--output-csv", default="attention_mode_matrix.csv")
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=MODE_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-len", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    validate_args(args)
    return args


def run_memory_worker(args) -> None:
    if args.worker_mode is None or args.worker_prompt_len is None:
        raise ValueError("Memory worker requires --worker-mode and --worker-prompt-len")

    backend = get_backend()
    prompts_np = make_prompts(args.batch_size, args.worker_prompt_len, args.n_vocab, args.seed)

    base_model = make_model(args, backend, use_flash_attn=False)
    flash_model = make_model(args, backend, use_flash_attn=True)
    copy_model_weights(base_model, flash_model, backend)
    mode = next(spec for spec in MODE_SPECS if spec.name == args.worker_mode)
    model = flash_model if mode.use_flash_attn else base_model
    _ = run_mode(mode, model, prompts_np, args, backend)
    sync_cuda()
    time.sleep(0.05)


def main():
    args = parse_args()
    if args.memory_worker:
        run_memory_worker(args)
        return

    os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    backend = get_backend()
    rows: List[Dict[str, object]] = []
    for prompt_len in args.prompt_lengths:
        rows.extend(benchmark_config(args, backend, prompt_len))
    rows = enrich_pairwise(rows)

    report = write_report(rows, args)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(report)
    csv_path = write_csv(rows, args.output_csv)
    print(report, end="")
    print(f"Saved report to {args.output_txt}")
    print(f"Saved csv to {csv_path}")


if __name__ == "__main__":
    main()
