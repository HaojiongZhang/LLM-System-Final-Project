#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import minitorch  # noqa: E402
from minitorch.tensor_functions import tensor_from_numpy  # noqa: E402
from minitorch.transformer import DecoderLM, MultiHeadAttention  # noqa: E402
from minitorch.paged_attention import BlockManager  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_AVAILABLE = True
except Exception:  # pragma: no cover
    pynvml = None
    _nvml_handle = None
    PYNVML_AVAILABLE = False


def cuda_sync() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_backend():
    return minitorch.TensorBackend(minitorch.CudaKernelOps)


def current_gpu_mem_mb():
    if PYNVML_AVAILABLE:
        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / (1024.0 * 1024.0)
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    return float("nan")


def current_process_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def current_process_rss_mb():
    rss_mb = _read_process_rss_mb(os.getpid())
    if np.isfinite(rss_mb):
        return rss_mb
    return current_process_peak_rss_mb()


def _read_process_rss_mb(pid):
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
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        if out:
            return float(out) / 1024.0
    except Exception:
        pass
    return float("nan")


def _measure_process_peak_gpu_mem_mb(proc):
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


def _measure_child_peaks(proc):
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


def sample_operation_peak(run_fn, sample_interval_s=0.001):
    baseline_rss_mb = current_process_rss_mb()
    baseline_gpu_mb = current_gpu_mem_mb()
    peak_rss_mb = baseline_rss_mb
    peak_gpu_mb = baseline_gpu_mb
    stop_event = threading.Event()

    def sampler():
        nonlocal peak_rss_mb, peak_gpu_mb
        while not stop_event.is_set():
            rss_mb = current_process_rss_mb()
            if np.isfinite(rss_mb):
                peak_rss_mb = max(peak_rss_mb, rss_mb)
            gpu_mb = current_gpu_mem_mb()
            if np.isfinite(gpu_mb):
                if not np.isfinite(peak_gpu_mb):
                    peak_gpu_mb = gpu_mb
                else:
                    peak_gpu_mb = max(peak_gpu_mb, gpu_mb)
            time.sleep(sample_interval_s)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        run_fn()
        cuda_sync()
    finally:
        stop_event.set()
        thread.join()

    delta_rss_mb = max(peak_rss_mb - baseline_rss_mb, 0.0) if np.isfinite(peak_rss_mb) and np.isfinite(baseline_rss_mb) else float("nan")
    delta_gpu_mb = max(peak_gpu_mb - baseline_gpu_mb, 0.0) if np.isfinite(peak_gpu_mb) and np.isfinite(baseline_gpu_mb) else float("nan")
    return {
        "delta_peak_rss_mb": delta_rss_mb,
        "delta_peak_gpu_mb": delta_gpu_mb,
    }


def warmup_operation(run_fn):
    # Exclude one-time allocator and kernel initialization from measured memory deltas.
    run_fn()
    cuda_sync()
    time.sleep(0.01)


def _spawn_memory_worker(args):
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    peaks = _measure_child_peaks(proc)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Memory worker failed\nstdout:\n{stdout}\nstderr:\n{stderr}")
    worker_metrics = {}
    stdout = stdout.strip()
    if stdout:
        try:
            worker_metrics = json.loads(stdout.splitlines()[-1])
        except Exception:
            worker_metrics = {}
    delta_peak_rss_mb = float(worker_metrics.get("delta_peak_rss_mb", float("nan")))
    delta_peak_gpu_mb = float(worker_metrics.get("delta_peak_gpu_mb", float("nan")))

    return {
        "delta_peak_rss_mb": delta_peak_rss_mb,
        "delta_peak_gpu_mb": delta_peak_gpu_mb,
        "raw_peak_gpu_mb": peaks["peak_gpu_mb"],
    }


def _fmt_float(value, suffix=""):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f}{suffix}"


def _memory_reduction_factor(baseline_peak, optimized_peak):
    if (
        not np.isfinite(baseline_peak)
        or not np.isfinite(optimized_peak)
        or optimized_peak <= 1e-9
    ):
        return float("nan")
    return baseline_peak / optimized_peak


def _memory_saved_pct(baseline_peak, optimized_peak):
    if not np.isfinite(baseline_peak) or baseline_peak <= 1e-9 or not np.isfinite(optimized_peak):
        return float("nan")
    return 100.0 * (baseline_peak - optimized_peak) / baseline_peak


def clone_parameters(dst_module, src_module, backend):
    dst_params = list(dst_module.parameters())
    src_params = list(src_module.parameters())
    assert len(dst_params) == len(src_params), "Parameter count mismatch"
    for dp, sp in zip(dst_params, src_params):
        arr = np.ascontiguousarray(sp.value.to_numpy(), dtype=np.float32)
        dp.update(tensor_from_numpy(arr, backend=backend))


def summarize_diff(a, b):
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    abs_diff = np.abs(a_np - b_np)
    return float(abs_diff.mean()), float(abs_diff.max())


def tensor_like_from_numpy(arr, backend, requires_grad=False):
    t = tensor_from_numpy(np.ascontiguousarray(arr.astype(np.float32)), backend=backend)
    t.requires_grad_(requires_grad)
    return t


def benchmark_call(fn, warmup=10, iters=20):
    for _ in range(warmup):
        fn()
    cuda_sync()

    times = []
    last = None
    for _ in range(iters):
        cuda_sync()
        t0 = time.perf_counter()
        last = fn()
        cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    return last, {
        "mean_s": float(times.mean()),
        "std_s": float(times.std()),
        "median_s": float(np.median(times)),
        "p95_s": float(np.percentile(times, 95)),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
    }


def benchmark_prepared_backward(prepare_fn, warmup=10, iters=20):
    for _ in range(warmup):
        loss, _x = prepare_fn()
        loss.backward()
    cuda_sync()

    times = []
    last_grad = None
    for _ in range(iters):
        loss, x = prepare_fn()
        cuda_sync()
        t0 = time.perf_counter()
        loss.backward()
        cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if x.grad is None:
            raise RuntimeError("Expected input gradient but got None")
        last_grad = x.grad

    times = np.array(times, dtype=np.float64)
    return last_grad, {
        "mean_s": float(times.mean()),
        "std_s": float(times.std()),
        "median_s": float(np.median(times)),
        "p95_s": float(np.percentile(times, 95)),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
    }


def build_attention_pair(n_embd, n_head, backend, causal=True):
    baseline = MultiHeadAttention(
        n_embd=n_embd,
        n_head=n_head,
        causal=causal,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=False,
    )
    flash = MultiHeadAttention(
        n_embd=n_embd,
        n_head=n_head,
        causal=causal,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=True,
    )
    clone_parameters(flash, baseline, backend)
    baseline.eval()
    flash.eval()
    return baseline, flash


def build_decoder_pair(n_vocab, n_embd, n_head, n_positions, backend):
    baseline = DecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=False,
    )
    integrated = DecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=True,
    )
    clone_parameters(integrated, baseline, backend)
    baseline.eval()
    integrated.eval()
    return baseline, integrated


def run_forward_once(module, x):
    q, kT, v = module.project_to_query_key_value(x)
    return module.self_attention(q, kT, v)


def run_forward_backward_once(module, x):
    x.zero_grad_()
    out = run_forward_once(module, x)
    loss = out.sum()
    loss.backward()
    if x.grad is None:
        raise RuntimeError("Expected input gradient but got None")
    return out, x.grad


def prepare_backward_only(module, x_np, backend):
    x = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    out = run_forward_once(module, x)
    loss = out.sum()
    return loss, x


def run_attention_memory_worker(batch_size, seq_len, n_embd, n_head, seed, mode, op):
    np.random.seed(seed)
    backend = make_backend()
    baseline, flash = build_attention_pair(n_embd, n_head, backend, causal=True)
    module = baseline if mode == "baseline" else flash
    x_np = np.random.randn(batch_size, seq_len, n_embd).astype(np.float32)

    if op == "forward":
        def run_op():
            x = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=False)
            _ = run_forward_once(module, x)
    elif op == "backward":
        def run_op():
            loss, _x = prepare_backward_only(module, x_np, backend)
            loss.backward()
    elif op == "fwbw":
        def run_op():
            x = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
            _ = run_forward_backward_once(module, x)
    else:
        raise ValueError(f"Unknown attention memory op: {op}")

    warmup_operation(run_op)
    metrics = sample_operation_peak(run_op)
    print(json.dumps(metrics))


def append_token_column(context, token_ids):
    return np.concatenate([context, token_ids[:, None].astype(np.int32)], axis=1)


def run_baseline_generation(model, prompts_np, new_tokens, backend):
    contexts = prompts_np.copy()
    generated = [[] for _ in range(prompts_np.shape[0])]
    step_last_logits = []

    logits = model(tensor_from_numpy(contexts, backend=backend))
    last_logits = np.ascontiguousarray(logits.to_numpy()[:, -1, :], dtype=np.float32)
    step_last_logits.append(last_logits.copy())
    next_tokens = last_logits.argmax(axis=-1).astype(np.int32)
    for b, tok in enumerate(next_tokens):
        generated[b].append(int(tok))
    contexts = append_token_column(contexts, next_tokens)

    t_decode0 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = model(tensor_from_numpy(contexts, backend=backend))
        last_logits = np.ascontiguousarray(logits.to_numpy()[:, -1, :], dtype=np.float32)
        step_last_logits.append(last_logits.copy())
        next_tokens = last_logits.argmax(axis=-1).astype(np.int32)
        for b, tok in enumerate(next_tokens):
            generated[b].append(int(tok))
        contexts = append_token_column(contexts, next_tokens)
    t_decode1 = time.perf_counter()

    decode_ms = (t_decode1 - t_decode0) * 1000.0
    return {
        "generated": generated,
        "step_last_logits": step_last_logits,
        "decode_ms": decode_ms,
        "decode_tok_s": (prompts_np.shape[0] * (new_tokens - 1)) / max((decode_ms / 1000.0), 1e-9),
    }


def run_integrated_generation(model, prompts_np, new_tokens, backend, block_size, n_head, n_embd):
    batch_size, prompt_len = prompts_np.shape
    num_blocks = batch_size * math.ceil((prompt_len + new_tokens + 1) / block_size) + 4
    bm = BlockManager(
        num_layers=4,
        num_blocks=num_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=n_embd // n_head,
        backend=backend,
    )
    seq_ids = list(range(batch_size))
    generated = [[] for _ in range(batch_size)]
    step_last_logits = []

    first_tokens = []
    first_logits = []
    for sid in seq_ids:
        prompt_t = tensor_from_numpy(prompts_np[sid:sid + 1], backend=backend)
        logits = model.prefill(prompt_t, sid, bm)
        last_logits = np.ascontiguousarray(logits.to_numpy()[0, -1, :], dtype=np.float32)
        first_logits.append(last_logits)
        tok = int(last_logits.argmax())
        generated[sid].append(tok)
        first_tokens.append(tok)
    step_last_logits.append(np.stack(first_logits, axis=0))

    token_ids_np = np.array(first_tokens, dtype=np.int32)[:, None]
    t_decode0 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = model.decode_step_batch(
            tensor_from_numpy(token_ids_np, backend=backend),
            seq_ids,
            bm,
        )
        last_logits = np.ascontiguousarray(logits.to_numpy()[:, 0, :], dtype=np.float32)
        step_last_logits.append(last_logits.copy())
        next_tokens = last_logits.argmax(axis=-1).astype(np.int32)
        for sid, tok in enumerate(next_tokens):
            generated[sid].append(int(tok))
        token_ids_np = next_tokens[:, None]
    t_decode1 = time.perf_counter()

    decode_ms = (t_decode1 - t_decode0) * 1000.0
    return {
        "generated": generated,
        "step_last_logits": step_last_logits,
        "decode_ms": decode_ms,
        "decode_tok_s": (batch_size * (new_tokens - 1)) / max((decode_ms / 1000.0), 1e-9),
    }


def run_inference_memory_worker(prompt_len, new_tokens, n_vocab, n_embd, n_head, n_positions, block_size, seed, mode):
    rng = np.random.default_rng(seed + prompt_len)
    backend = make_backend()
    baseline, flash = build_decoder_pair(n_vocab, n_embd, n_head, n_positions, backend)
    prompts_np = rng.integers(0, n_vocab, size=(1, prompt_len), dtype=np.int32)

    if mode == "baseline":
        def run_op():
            _ = run_baseline_generation(baseline, prompts_np, new_tokens, backend)
    elif mode == "flash":
        def run_op():
            _ = run_baseline_generation(flash, prompts_np, new_tokens, backend)
    elif mode == "paged":
        def run_op():
            _ = run_integrated_generation(baseline, prompts_np, new_tokens, backend, block_size, n_head, n_embd)
    elif mode == "integrated":
        def run_op():
            _ = run_integrated_generation(flash, prompts_np, new_tokens, backend, block_size, n_head, n_embd)
    else:
        raise ValueError(f"Unknown inference memory mode: {mode}")

    warmup_operation(run_op)
    metrics = sample_operation_peak(run_op)
    print(json.dumps(metrics))


def measure_attention_memory_case(batch_size, seq_len, n_embd, n_head, seed, repeats):
    results = {}
    for op in ("forward", "backward", "fwbw"):
        baseline_runs = []
        flash_runs = []
        for _ in range(repeats):
            common = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--memory-worker",
                "--worker-kind",
                "attention",
                "--worker-op",
                op,
                "--worker-seq-len",
                str(seq_len),
                "--bs",
                str(batch_size),
                "--embd",
                str(n_embd),
                "--heads",
                str(n_head),
                "--seed",
                str(seed),
            ]
            baseline_runs.append(_spawn_memory_worker(common + ["--worker-mode", "baseline"]))
            flash_runs.append(_spawn_memory_worker(common + ["--worker-mode", "flash"]))

        baseline_rss = float(np.median([row["delta_peak_rss_mb"] for row in baseline_runs]))
        flash_rss = float(np.median([row["delta_peak_rss_mb"] for row in flash_runs]))
        baseline_gpu = float(np.median([row["delta_peak_gpu_mb"] for row in baseline_runs]))
        flash_gpu = float(np.median([row["delta_peak_gpu_mb"] for row in flash_runs]))

        results[f"{op}_baseline_delta_rss_mb"] = baseline_rss
        results[f"{op}_flash_delta_rss_mb"] = flash_rss
        results[f"{op}_rss_reduction_factor"] = _memory_reduction_factor(baseline_rss, flash_rss)
        results[f"{op}_rss_saved_pct"] = _memory_saved_pct(baseline_rss, flash_rss)
        results[f"{op}_baseline_delta_gpu_mb"] = baseline_gpu
        results[f"{op}_flash_delta_gpu_mb"] = flash_gpu
        results[f"{op}_gpu_reduction_factor"] = _memory_reduction_factor(baseline_gpu, flash_gpu)
        results[f"{op}_gpu_saved_pct"] = _memory_saved_pct(baseline_gpu, flash_gpu)

    return results


def measure_inference_memory_case(prompt_len, new_tokens, n_vocab, n_embd, n_head, n_positions, block_size, seed, repeats):
    baseline_runs = []
    flash_runs = []
    paged_runs = []
    integrated_runs = []
    for _ in range(repeats):
        common = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--memory-worker",
            "--worker-kind",
            "inference",
            "--worker-prompt-len",
            str(prompt_len),
            "--new-tokens",
            str(new_tokens),
            "--n-vocab",
            str(n_vocab),
            "--embd",
            str(n_embd),
            "--heads",
            str(n_head),
            "--n-positions",
            str(n_positions),
            "--block-size",
            str(block_size),
            "--seed",
            str(seed),
        ]
        baseline_runs.append(_spawn_memory_worker(common + ["--worker-mode", "baseline"]))
        flash_runs.append(_spawn_memory_worker(common + ["--worker-mode", "flash"]))
        paged_runs.append(_spawn_memory_worker(common + ["--worker-mode", "paged"]))
        integrated_runs.append(_spawn_memory_worker(common + ["--worker-mode", "integrated"]))

    baseline_rss = float(np.median([row["delta_peak_rss_mb"] for row in baseline_runs]))
    flash_rss = float(np.median([row["delta_peak_rss_mb"] for row in flash_runs]))
    paged_rss = float(np.median([row["delta_peak_rss_mb"] for row in paged_runs]))
    integrated_rss = float(np.median([row["delta_peak_rss_mb"] for row in integrated_runs]))
    baseline_gpu = float(np.median([row["delta_peak_gpu_mb"] for row in baseline_runs]))
    flash_gpu = float(np.median([row["delta_peak_gpu_mb"] for row in flash_runs]))
    paged_gpu = float(np.median([row["delta_peak_gpu_mb"] for row in paged_runs]))
    integrated_gpu = float(np.median([row["delta_peak_gpu_mb"] for row in integrated_runs]))

    return {
        "baseline_delta_rss_mb": baseline_rss,
        "flash_only_delta_rss_mb": flash_rss,
        "paged_only_delta_rss_mb": paged_rss,
        "integrated_delta_rss_mb": integrated_rss,
        "flash_only_rss_reduction_factor": _memory_reduction_factor(baseline_rss, flash_rss),
        "paged_only_rss_reduction_factor": _memory_reduction_factor(baseline_rss, paged_rss),
        "integrated_rss_reduction_factor": _memory_reduction_factor(baseline_rss, integrated_rss),
        "flash_only_rss_saved_pct": _memory_saved_pct(baseline_rss, flash_rss),
        "paged_only_rss_saved_pct": _memory_saved_pct(baseline_rss, paged_rss),
        "integrated_rss_saved_pct": _memory_saved_pct(baseline_rss, integrated_rss),
        "baseline_delta_gpu_mb": baseline_gpu,
        "flash_only_delta_gpu_mb": flash_gpu,
        "paged_only_delta_gpu_mb": paged_gpu,
        "integrated_delta_gpu_mb": integrated_gpu,
        "flash_only_gpu_reduction_factor": _memory_reduction_factor(baseline_gpu, flash_gpu),
        "paged_only_gpu_reduction_factor": _memory_reduction_factor(baseline_gpu, paged_gpu),
        "integrated_gpu_reduction_factor": _memory_reduction_factor(baseline_gpu, integrated_gpu),
        "flash_only_gpu_saved_pct": _memory_saved_pct(baseline_gpu, flash_gpu),
        "paged_only_gpu_saved_pct": _memory_saved_pct(baseline_gpu, paged_gpu),
        "integrated_gpu_saved_pct": _memory_saved_pct(baseline_gpu, integrated_gpu),
    }


def max_step_logit_diff(a_steps, b_steps):
    max_diff = 0.0
    for a, b in zip(a_steps, b_steps):
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    return max_diff


def run_attention_case(batch_size, seq_len, n_embd, n_head, warmup, iters, memory_repeats, causal=True, seed=0):
    np.random.seed(seed)
    backend = make_backend()
    baseline, flash = build_attention_pair(n_embd, n_head, backend, causal=causal)
    x_np = np.random.randn(batch_size, seq_len, n_embd).astype(np.float32)

    x_base = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    x_flash = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    out_fwd_base = run_forward_once(baseline, x_base)
    out_fwd_flash = run_forward_once(flash, x_flash)
    out_fwd_mean_diff, out_fwd_max_diff = summarize_diff(out_fwd_base, out_fwd_flash)

    x_base_bwd = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    x_flash_bwd = tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
    out_base, grad_base = run_forward_backward_once(baseline, x_base_bwd)
    out_flash, grad_flash = run_forward_backward_once(flash, x_flash_bwd)
    out_mean_diff, out_max_diff = summarize_diff(out_base, out_flash)
    grad_mean_diff, grad_max_diff = summarize_diff(grad_base, grad_flash)

    _, forward_base_stats = benchmark_call(
        lambda: run_forward_once(
            baseline, tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=False)
        ),
        warmup=warmup,
        iters=iters,
    )
    _, forward_flash_stats = benchmark_call(
        lambda: run_forward_once(
            flash, tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=False)
        ),
        warmup=warmup,
        iters=iters,
    )

    _, backward_base_stats = benchmark_prepared_backward(
        lambda: prepare_backward_only(baseline, x_np, backend),
        warmup=warmup,
        iters=iters,
    )
    _, backward_flash_stats = benchmark_prepared_backward(
        lambda: prepare_backward_only(flash, x_np, backend),
        warmup=warmup,
        iters=iters,
    )

    (_, _), fwbw_base_stats = benchmark_call(
        lambda: run_forward_backward_once(
            baseline, tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
        ),
        warmup=warmup,
        iters=iters,
    )
    (_, _), fwbw_flash_stats = benchmark_call(
        lambda: run_forward_backward_once(
            flash, tensor_like_from_numpy(x_np.copy(), backend=backend, requires_grad=True)
        ),
        warmup=warmup,
        iters=iters,
    )

    tokens = batch_size * seq_len
    row = {
        "section": "attention",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": n_embd // n_head,
        "forward_baseline_mean_s": forward_base_stats["mean_s"],
        "forward_flash_mean_s": forward_flash_stats["mean_s"],
        "forward_speedup_mean": forward_base_stats["mean_s"] / max(forward_flash_stats["mean_s"], 1e-12),
        "forward_baseline_tok_s": tokens / max(forward_base_stats["mean_s"], 1e-12),
        "forward_flash_tok_s": tokens / max(forward_flash_stats["mean_s"], 1e-12),
        "backward_baseline_mean_s": backward_base_stats["mean_s"],
        "backward_flash_mean_s": backward_flash_stats["mean_s"],
        "backward_speedup_mean": backward_base_stats["mean_s"] / max(backward_flash_stats["mean_s"], 1e-12),
        "fwbw_baseline_mean_s": fwbw_base_stats["mean_s"],
        "fwbw_flash_mean_s": fwbw_flash_stats["mean_s"],
        "fwbw_speedup_mean": fwbw_base_stats["mean_s"] / max(fwbw_flash_stats["mean_s"], 1e-12),
        "output_fwd_mean_abs_diff": out_fwd_mean_diff,
        "output_fwd_max_abs_diff": out_fwd_max_diff,
        "output_mean_abs_diff": out_mean_diff,
        "output_max_abs_diff": out_max_diff,
        "input_grad_mean_abs_diff": grad_mean_diff,
        "input_grad_max_abs_diff": grad_max_diff,
    }
    row.update(
        measure_attention_memory_case(
            batch_size=batch_size,
            seq_len=seq_len,
            n_embd=n_embd,
            n_head=n_head,
            seed=seed,
            repeats=memory_repeats,
        )
    )
    return row


def run_inference_case(prompt_len, new_tokens, n_vocab, n_embd, n_head, n_positions, block_size, warmup, iters, memory_repeats, seed=0):
    rng = np.random.default_rng(seed + prompt_len)
    backend = make_backend()
    baseline, flash = build_decoder_pair(n_vocab, n_embd, n_head, n_positions, backend)
    prompts_np = rng.integers(0, n_vocab, size=(1, prompt_len), dtype=np.int32)

    base_once = run_baseline_generation(baseline, prompts_np, new_tokens, backend)
    flash_once = run_baseline_generation(flash, prompts_np, new_tokens, backend)
    paged_once = run_integrated_generation(baseline, prompts_np, new_tokens, backend, block_size, n_head, n_embd)
    int_once = run_integrated_generation(flash, prompts_np, new_tokens, backend, block_size, n_head, n_embd)

    _, base_stats = benchmark_call(
        lambda: run_baseline_generation(baseline, prompts_np, new_tokens, backend),
        warmup=warmup,
        iters=iters,
    )
    _, flash_stats = benchmark_call(
        lambda: run_baseline_generation(flash, prompts_np, new_tokens, backend),
        warmup=warmup,
        iters=iters,
    )
    _, paged_stats = benchmark_call(
        lambda: run_integrated_generation(baseline, prompts_np, new_tokens, backend, block_size, n_head, n_embd),
        warmup=warmup,
        iters=iters,
    )
    _, int_stats = benchmark_call(
        lambda: run_integrated_generation(flash, prompts_np, new_tokens, backend, block_size, n_head, n_embd),
        warmup=warmup,
        iters=iters,
    )

    row = {
        "section": "inference",
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": n_embd // n_head,
        "baseline_total_mean_s": base_stats["mean_s"],
        "flash_only_total_mean_s": flash_stats["mean_s"],
        "paged_only_total_mean_s": paged_stats["mean_s"],
        "integrated_total_mean_s": int_stats["mean_s"],
        "flash_only_speedup_mean": base_stats["mean_s"] / max(flash_stats["mean_s"], 1e-12),
        "paged_only_speedup_mean": base_stats["mean_s"] / max(paged_stats["mean_s"], 1e-12),
        "integrated_speedup_mean": base_stats["mean_s"] / max(int_stats["mean_s"], 1e-12),
        "baseline_decode_tok_s": base_once["decode_tok_s"],
        "flash_only_decode_tok_s": flash_once["decode_tok_s"],
        "paged_only_decode_tok_s": paged_once["decode_tok_s"],
        "integrated_decode_tok_s": int_once["decode_tok_s"],
        "flash_only_tokens_match": base_once["generated"] == flash_once["generated"],
        "paged_only_tokens_match": base_once["generated"] == paged_once["generated"],
        "integrated_tokens_match": base_once["generated"] == int_once["generated"],
        "flash_only_max_step_logit_diff": max_step_logit_diff(base_once["step_last_logits"], flash_once["step_last_logits"]),
        "paged_only_max_step_logit_diff": max_step_logit_diff(base_once["step_last_logits"], paged_once["step_last_logits"]),
        "integrated_max_step_logit_diff": max_step_logit_diff(base_once["step_last_logits"], int_once["step_last_logits"]),
    }
    row.update(
        measure_inference_memory_case(
            prompt_len=prompt_len,
            new_tokens=new_tokens,
            n_vocab=n_vocab,
            n_embd=n_embd,
            n_head=n_head,
            n_positions=n_positions,
            block_size=block_size,
            seed=seed,
            repeats=memory_repeats,
        )
    )
    return row


def print_attention_result(r):
    print("=" * 100)
    print(
        f"ATTN bs={r['batch_size']:>3} seq={r['seq_len']:>4} "
        f"embd={r['n_embd']:>4} heads={r['n_head']:>2} head_dim={r['head_dim']:>3}"
    )
    print(
        f"forward  : base={r['forward_baseline_mean_s']:.6f}s "
        f"flash={r['forward_flash_mean_s']:.6f}s "
        f"speedup={r['forward_speedup_mean']:.3f}x"
    )
    print(
        f"backward : base={r['backward_baseline_mean_s']:.6f}s "
        f"flash={r['backward_flash_mean_s']:.6f}s "
        f"speedup={r['backward_speedup_mean']:.3f}x"
    )
    print(
        f"fwd+bwd  : base={r['fwbw_baseline_mean_s']:.6f}s "
        f"flash={r['fwbw_flash_mean_s']:.6f}s "
        f"speedup={r['fwbw_speedup_mean']:.3f}x"
    )
    print(
        f"fwd diff : mean_abs={r['output_fwd_mean_abs_diff']:.3e} "
        f"max_abs={r['output_fwd_max_abs_diff']:.3e}"
    )
    print(
        f"grad diff: mean_abs={r['input_grad_mean_abs_diff']:.3e} "
        f"max_abs={r['input_grad_max_abs_diff']:.3e}"
    )
    for op, label in (("forward", "fwd"), ("backward", "bwd"), ("fwbw", "fwd+bwd")):
        print(
            f"mem {label:<6}: delta_rss base={_fmt_float(r[f'{op}_baseline_delta_rss_mb'], ' MB')} "
            f"flash={_fmt_float(r[f'{op}_flash_delta_rss_mb'], ' MB')} "
            f"factor={_fmt_float(r[f'{op}_rss_reduction_factor'], 'x')} "
            f"saved={_fmt_float(r[f'{op}_rss_saved_pct'], '%')}"
        )
        if PYNVML_AVAILABLE:
            print(
                f"           delta_gpu base={_fmt_float(r[f'{op}_baseline_delta_gpu_mb'], ' MB')} "
                f"flash={_fmt_float(r[f'{op}_flash_delta_gpu_mb'], ' MB')} "
                f"factor={_fmt_float(r[f'{op}_gpu_reduction_factor'], 'x')} "
                f"saved={_fmt_float(r[f'{op}_gpu_saved_pct'], '%')}"
            )


def print_inference_result(r):
    print("=" * 100)
    print(
        f"INFER prompt={r['prompt_len']:>4} new_tokens={r['new_tokens']:>3} "
        f"embd={r['n_embd']:>4} heads={r['n_head']:>2}"
    )
    print(
        f"none total={r['baseline_total_mean_s']:.6f}s "
        f"FA-only={r['flash_only_total_mean_s']:.6f}s "
        f"PA-only={r['paged_only_total_mean_s']:.6f}s "
        f"FA+PA={r['integrated_total_mean_s']:.6f}s"
    )
    print(
        f"speedups   : FA-only={r['flash_only_speedup_mean']:.3f}x "
        f"PA-only={r['paged_only_speedup_mean']:.3f}x "
        f"FA+PA={r['integrated_speedup_mean']:.3f}x"
    )
    print(
        f"decode tok/s: none={r['baseline_decode_tok_s']:.2f} "
        f"FA-only={r['flash_only_decode_tok_s']:.2f} "
        f"PA-only={r['paged_only_decode_tok_s']:.2f} "
        f"FA+PA={r['integrated_decode_tok_s']:.2f}"
    )
    print(
        f"token match : FA-only={r['flash_only_tokens_match']} "
        f"PA-only={r['paged_only_tokens_match']} "
        f"FA+PA={r['integrated_tokens_match']}"
    )
    print(
        f"logit diff  : FA-only={r['flash_only_max_step_logit_diff']:.3e} "
        f"PA-only={r['paged_only_max_step_logit_diff']:.3e} "
        f"FA+PA={r['integrated_max_step_logit_diff']:.3e}"
    )
    print(
        f"mem delta_rss : none={_fmt_float(r['baseline_delta_rss_mb'], ' MB')} "
        f"FA-only={_fmt_float(r['flash_only_delta_rss_mb'], ' MB')} "
        f"PA-only={_fmt_float(r['paged_only_delta_rss_mb'], ' MB')} "
        f"FA+PA={_fmt_float(r['integrated_delta_rss_mb'], ' MB')}"
    )
    print(
        f"rss saved    : FA-only { _fmt_float(r['flash_only_rss_reduction_factor'], 'x') } / { _fmt_float(r['flash_only_rss_saved_pct'], '%') }  "
        f"PA-only { _fmt_float(r['paged_only_rss_reduction_factor'], 'x') } / { _fmt_float(r['paged_only_rss_saved_pct'], '%') }  "
        f"FA+PA { _fmt_float(r['integrated_rss_reduction_factor'], 'x') } / { _fmt_float(r['integrated_rss_saved_pct'], '%') }"
    )
    if PYNVML_AVAILABLE:
        print(
            f"mem delta_gpu : none={_fmt_float(r['baseline_delta_gpu_mb'], ' MB')} "
            f"FA-only={_fmt_float(r['flash_only_delta_gpu_mb'], ' MB')} "
            f"PA-only={_fmt_float(r['paged_only_delta_gpu_mb'], ' MB')} "
            f"FA+PA={_fmt_float(r['integrated_delta_gpu_mb'], ' MB')}"
        )
        print(
            f"gpu saved    : FA-only { _fmt_float(r['flash_only_gpu_reduction_factor'], 'x') } / { _fmt_float(r['flash_only_gpu_saved_pct'], '%') }  "
            f"PA-only { _fmt_float(r['paged_only_gpu_reduction_factor'], 'x') } / { _fmt_float(r['paged_only_gpu_saved_pct'], '%') }  "
            f"FA+PA { _fmt_float(r['integrated_gpu_reduction_factor'], 'x') } / { _fmt_float(r['integrated_gpu_saved_pct'], '%') }"
        )


def write_csv(results, path):
    all_fields = set()
    for row in results:
        all_fields.update(row.keys())
    fieldnames = sorted(all_fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--embd", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--seqs", nargs="+", default=None, help="Attention seq lengths, e.g. 64 128 256")
    parser.add_argument("--prompt-seqs", nargs="+", default=None, help="Inference prompt lengths, default: same as --seqs")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-positions", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--csv", type=str, default="flash_fwbw_paged_results.csv")
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-kind", choices=["attention", "inference"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=["baseline", "flash", "paged", "integrated"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-op", choices=["forward", "backward", "fwbw"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-seq-len", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-len", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.memory_worker:
        if args.worker_kind == "attention":
            if args.worker_mode not in {"baseline", "flash"} or args.worker_op is None or args.worker_seq_len is None:
                raise ValueError("Attention memory worker requires --worker-mode, --worker-op, and --worker-seq-len")
            run_attention_memory_worker(
                batch_size=args.bs,
                seq_len=args.worker_seq_len,
                n_embd=args.embd,
                n_head=args.heads,
                seed=args.seed,
                mode=args.worker_mode,
                op=args.worker_op,
            )
            return
        if args.worker_kind == "inference":
            if args.worker_mode not in {"baseline", "flash", "paged", "integrated"} or args.worker_prompt_len is None:
                raise ValueError("Inference memory worker requires --worker-mode and --worker-prompt-len")
            run_inference_memory_worker(
                prompt_len=args.worker_prompt_len,
                new_tokens=args.new_tokens,
                n_vocab=args.n_vocab,
                n_embd=args.embd,
                n_head=args.heads,
                n_positions=args.n_positions,
                block_size=args.block_size,
                seed=args.seed,
                mode=args.worker_mode,
            )
            return
        raise ValueError("Memory worker requires --worker-kind")

    if args.seqs is None:
        raise ValueError("Normal benchmark run requires --seqs")

    seqs = [int(x) for x in args.seqs]
    prompt_seqs = [int(x) for x in (args.prompt_seqs if args.prompt_seqs is not None else args.seqs)]
    results = []

    print(
        f"Running attention cases with bs={args.bs}, embd={args.embd}, heads={args.heads}, seqs={seqs}"
    )
    for seq_len in seqs:
        r = run_attention_case(
            batch_size=args.bs,
            seq_len=seq_len,
            n_embd=args.embd,
            n_head=args.heads,
            warmup=args.warmup,
            iters=args.iters,
            memory_repeats=args.memory_repeats,
            seed=args.seed,
            causal=True,
        )
        results.append(r)
        print_attention_result(r)

    print("\n" + "#" * 100)
    print("Inference")
    print("#" * 100)
    for prompt_len in prompt_seqs:
        r = run_inference_case(
            prompt_len=prompt_len,
            new_tokens=args.new_tokens,
            n_vocab=args.n_vocab,
            n_embd=args.embd,
            n_head=args.heads,
            n_positions=args.n_positions,
            block_size=args.block_size,
            warmup=args.warmup,
            iters=args.iters,
            memory_repeats=args.memory_repeats,
            seed=args.seed,
        )
        results.append(r)
        print_inference_result(r)

    csv_path = Path(args.csv)
    write_csv(results, csv_path)
    print(f"\nWrote CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
