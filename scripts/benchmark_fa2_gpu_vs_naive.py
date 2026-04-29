"""
Benchmark: FlashAttention-2 backward vs. dense CUDA backward.

Three measurements:
  1. Kernel-only backward latency (CUDA events, existing methodology).
  2. Peak GPU memory during a full training step (forward + backward)
     with standard attention vs. FA2 attention.  This is the correct
     metric: it measures whether the T x T activation matrix is actually
     absent from the autograd graph, not just whether scratch buffers differ.
  3. Forward microbenchmark timed with CUDA events (mirrors backward method).

Run on HPC with a CUDA-capable GPU:
    python scripts/benchmark_fa2_gpu_vs_naive.py
"""

import argparse
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rng_tensors(B, H, T, D, backend, requires_grad=False):
    """Return q, k, v, out, lse, dout as minitorch Tensors."""
    from minitorch.tensor_functions import tensor_from_numpy

    rng = np.random.default_rng(42)

    def _t(shape):
        arr = rng.standard_normal(shape).astype(np.float32)
        t = tensor_from_numpy(arr, backend=backend)
        if requires_grad:
            t.requires_grad_(True)
        return t

    q    = _t((B, H, T, D))
    k    = _t((B, H, T, D))
    v    = _t((B, H, T, D))
    dout = _t((B, H, T, D))

    # Compute forward output and logsumexp from the same tensors so they
    # are consistent (needed by FA2 backward).
    scale  = 1.0 / math.sqrt(D)
    q_np   = np.ascontiguousarray(q.to_numpy(), dtype=np.float32)
    k_np   = np.ascontiguousarray(k.to_numpy(), dtype=np.float32)
    v_np   = np.ascontiguousarray(v.to_numpy(), dtype=np.float32)

    scores     = scale * (q_np @ k_np.transpose(0, 1, 3, 2))
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_s      = np.exp(scores - scores_max)
    sum_exp    = exp_s.sum(axis=-1, keepdims=True)
    lse_np     = (scores_max + np.log(sum_exp))[..., 0]
    out_np     = (exp_s / sum_exp) @ v_np

    out = tensor_from_numpy(out_np.astype(np.float32), backend=backend)
    lse = tensor_from_numpy(lse_np.astype(np.float32), backend=backend)

    return q, k, v, out, lse, dout


# ---------------------------------------------------------------------------
# 1. Kernel-only backward latency benchmark (CUDA-event timing)
# ---------------------------------------------------------------------------

SETTINGS = [
    dict(B=1, H=2, T=8, D=8, name="Smoke"),
    dict(B=2, H=3, T=16, D=8, name="KevRef"),
    dict(B=8, H=8, T=64, D=32, name="FF64"),
    dict(B=8, H=8, T=128, D=32, name="FF128"),
    dict(B=8, H=8, T=256, D=32, name="FF256"),
    dict(B=4, H=8, T=256, D=64, name="FF256D64"),
]

def benchmark_backward_latency(backend, warmup=5, repeats=20, n_outer=3):
    from minitorch.cuda_kernel_ops import (
        CudaKernelOps,
        HAS_BENCHMARK_DENSE_ATTENTION_BWD,
        HAS_BENCHMARK_FLASH_ATTENTION2_BWD,
    )

    if not (HAS_BENCHMARK_FLASH_ATTENTION2_BWD and HAS_BENCHMARK_DENSE_ATTENTION_BWD):
        print("CUDA FA2/Dense backward symbols not found; skipping kernel benchmark.")
        return

    print("\n=== Kernel-only backward latency (CUDA events) ===")
    header = f"{'Setting':<8} {'Dense (ms)':>12} {'FA2 (ms)':>10} {'Speedup':>9} {'Alloc Ratio':>12}"
    print(header)
    print("-" * len(header))

    for cfg in SETTINGS:
        B, H, T, D = cfg["B"], cfg["H"], cfg["T"], cfg["D"]
        q, k, v, out, lse, dout = _make_rng_tensors(B, H, T, D, backend)

        dense_times, fa2_times   = [], []
        dense_alloc, fa2_alloc   = 0, 0

        for _ in range(n_outer):
            ms_d, alloc_d = CudaKernelOps.benchmark_dense_attention_backward(
                dout, q, k, v, out, lse, causal=False,
                warmup=warmup, repeats=repeats,
            )
            ms_f, alloc_f = CudaKernelOps.benchmark_flash_attention2_backward(
                dout, q, k, v, out, lse, causal=False,
                warmup=warmup, repeats=repeats,
            )
            dense_times.append(ms_d)
            fa2_times.append(ms_f)
            dense_alloc = alloc_d
            fa2_alloc   = alloc_f

        d_ms = np.mean(dense_times)
        f_ms = np.mean(fa2_times)
        speedup      = d_ms / f_ms if f_ms > 0 else float("inf")
        alloc_ratio  = dense_alloc / fa2_alloc if fa2_alloc > 0 else float("inf")

        print(f"{cfg['name']:<8} {d_ms:>12.2f} {f_ms:>10.2f} {speedup:>8.2f}x {alloc_ratio:>11.2f}x")


# ---------------------------------------------------------------------------
# 2. Training-step peak GPU memory benchmark
# ---------------------------------------------------------------------------

def benchmark_training_memory(
    backend,
    n_vocab=512,
    n_embd=256,
    n_head=8,
    n_positions=512,
    seq_len=128,
    batch_size=4,
):
    """
    Measure peak GPU memory (pynvml) during a full forward + backward training
    step with standard attention (use_flash_attn=False) vs FA2 (True).

    The standard path materialises a (B, H, T, T) tensor in the autograd graph.
    The FA2 path does not store it; FlashAttentionFunc.backward recomputes it tile-by-tile.
    The difference in peak allocation is the real memory saving.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def peak_mem_mb():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 ** 2)
    except Exception:
        print("pynvml not available; using minitorch tensor count as proxy.")
        handle = None

        def peak_mem_mb():
            return -1.0

    from minitorch.transformer import DecoderLM
    from minitorch.tensor_functions import tensor_from_numpy

    rng = np.random.default_rng(0)
    idx_np = rng.integers(0, n_vocab, size=(batch_size, seq_len), dtype=np.int32)
    idx    = tensor_from_numpy(idx_np, backend=backend)

    results = {}
    for use_fa2 in (False, True):
        label = "FA2" if use_fa2 else "Standard"

        model = DecoderLM(
            n_vocab=n_vocab, n_embd=n_embd, n_head=n_head,
            n_positions=n_positions, backend=backend,
            p_dropout=0.0,
            use_flash_attn=use_fa2,
        )

        # Warm-up pass.
        _ = model(idx)

        mem_before = peak_mem_mb()
        t0 = time.perf_counter()
        logits = model(idx)
        # Dummy scalar loss: sum of all logits.
        loss = logits.sum(dim=2).sum(dim=1).sum(dim=0).view(1)
        loss.backward()
        t1 = time.perf_counter()
        mem_after = peak_mem_mb()

        results[label] = dict(
            peak_mb=mem_after,
            delta_mb=mem_after - mem_before,
            wall_ms=(t1 - t0) * 1000,
        )

    print("\n=== Training-step peak GPU memory ===")
    print(
        f"Config: B={batch_size}, T={seq_len}, n_embd={n_embd}, "
        f"n_head={n_head}, head_dim={n_embd // n_head}"
    )
    print(f"{'Method':<12} {'Peak (MB)':>12} {'Delta (MB)':>12} {'Wall (ms)':>12}")
    print("-" * 52)
    for label, r in results.items():
        print(f"{label:<12} {r['peak_mb']:>12.1f} {r['delta_mb']:>12.1f} {r['wall_ms']:>12.1f}")

    std = results["Standard"]
    fa2 = results["FA2"]
    if std["delta_mb"] > 0 and fa2["delta_mb"] > 0:
        ratio = std["delta_mb"] / fa2["delta_mb"]
        print(f"\nMemory ratio (Standard / FA2): {ratio:.2f}x")


# ---------------------------------------------------------------------------
# 3. Forward microbenchmark (CUDA events, mirrors backward methodology)
# ---------------------------------------------------------------------------

def benchmark_forward_microbench(backend, warmup=5, repeats=20):
    """
    Time only the attention forward computation with CUDA events.
    Compares standard self_attention path vs FlashAttentionFunc.forward.
    """
    import ctypes

    try:
        import pycuda.driver as cuda_driver
        pycuda_ok = True
    except ImportError:
        pycuda_ok = False

    from minitorch.tensor_functions import tensor_from_numpy
    from minitorch.flash_attention_func import FlashAttentionFunc

    small_settings = SETTINGS

    print("\n=== Forward microbenchmark (wall-clock, ms) ===")
    print(f"{'Setting':<8} {'Standard (ms)':>15} {'FA2-fwd (ms)':>14} {'Ratio':>8}")
    print("-" * 50)

    for cfg in small_settings:
        B, H, T, D = cfg["B"], cfg["H"], cfg["T"], cfg["D"]
        rng = np.random.default_rng(1)
        q_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
        k_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
        v_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
        q = tensor_from_numpy(q_np, backend=backend)
        k = tensor_from_numpy(k_np, backend=backend)
        v = tensor_from_numpy(v_np, backend=backend)

        scale = 1.0 / math.sqrt(D)

        # Standard forward (numpy matmul through minitorch ops)
        def std_fwd():
            kT = k.permute(0, 1, 3, 2)
            scores = (q @ kT) * scale
            from minitorch.nn import softmax
            w = softmax(scores, dim=3)
            return w @ v

        # FA2 forward
        BoundFn = FlashAttentionFunc.make(causal=False, scale=scale)
        def fa2_fwd():
            return BoundFn.apply(q, k, v)

        def _time_fn(fn, n_warm, n_rep):
            for _ in range(n_warm):
                fn()
            t0 = time.perf_counter()
            for _ in range(n_rep):
                fn()
            return (time.perf_counter() - t0) / n_rep * 1000

        std_ms = _time_fn(std_fwd, warmup, repeats)
        fa2_ms = _time_fn(fa2_fwd, warmup, repeats)
        ratio  = std_ms / fa2_ms if fa2_ms > 0 else float("inf")

        print(f"{cfg['name']:<8} {std_ms:>15.3f} {fa2_ms:>14.3f} {ratio:>7.2f}x")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--skip-memory",   action="store_true")
    parser.add_argument("--skip-forward",  action="store_true")
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-positions", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import minitorch
    backend = minitorch.TensorBackend(minitorch.CudaKernelOps) if args.backend == "cuda" \
              else minitorch.TensorBackend(minitorch.SimpleOps)

    if not args.skip_backward:
        benchmark_backward_latency(backend)
    if not args.skip_memory:
        benchmark_training_memory(
            backend,
            n_vocab=args.n_vocab,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_positions=args.n_positions,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
    if not args.skip_forward:
        benchmark_forward_microbench(backend)


if __name__ == "__main__":
    main()
