#!/usr/bin/env python3
"""
Benchmark paged vs. standard attention.
Measures: correctness, latency, memory usage, throughput.

Latency baseline: contiguous KV cache (not no-cache).
  Both paths store K/V after prefill and run single-token decode.
  The only difference is how K/V is stored and retrieved:
    - Contiguous: flat tensor, direct index
    - Paged:      block table indirection

Memory baseline: N concurrent sequences, each at a different length.
  Standard pre-allocates max_seq_len per sequence (typical naive implementation).
  Paged allocates one shared pool sized for actual tokens only.
  This is the scenario where paged attention's memory savings are observable.

Throughput baseline: increasing batch sizes until standard OOMs.
  Paged fits more sequences in the same GPU memory -> higher batch size
  -> more tokens decoded per second. This is the primary vLLM metric.
"""

import numpy as np
import time
import torch
import argparse
from statistics import stdev

import minitorch
from minitorch.paged_attention import BlockManager
from minitorch.tensor import tensor_from_numpy

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


def get_gpu_mem_mb():
    """Return current total GPU memory used in MB."""
    if PYNVML_AVAILABLE:
        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / 1024 / 1024
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_backend():
    return minitorch.TensorBackend(minitorch.CudaKernelOps)


# ============================================================================
# CONTIGUOUS KV CACHE HELPERS
# Mirrors the paged path exactly: same numpy/CPU storage, same data flow.
# Only difference: no block table — direct index into a flat buffer.
# ============================================================================

def contiguous_prefill(attn, x_context):
    """
    Run a prefill pass and return K, V as contiguous numpy arrays.

    Returns:
        k_cache: (1, n_head, ctx_len, head_dim)
        v_cache: (1, n_head, ctx_len, head_dim)
    """
    _, kT, v = attn.project_to_query_key_value(x_context)
    # kT: (1, n_head, head_dim, ctx_len) -> transpose last two dims
    k_cache = kT.to_numpy().transpose(0, 1, 3, 2)
    v_cache = v.to_numpy()
    return k_cache, v_cache


def contiguous_decode(attn, x_token, k_cache, v_cache, backend):
    """
    Decode one new token using a contiguous KV cache.
    Appends new K/V to cache, then runs attention over the full history.

    Args:
        k_cache: (1, n_head, ctx_len, head_dim) numpy
        v_cache: (1, n_head, ctx_len, head_dim) numpy

    Returns:
        output tensor (1, 1, n_embd)
    """
    n_embd = attn.n_embd

    q, kT_new, v_new = attn.project_to_query_key_value(x_token)
    k_new = kT_new.to_numpy().transpose(0, 1, 3, 2)  # (1, n_head, 1, head_dim)
    v_new_np = v_new.to_numpy()                        # (1, n_head, 1, head_dim)

    # Append new token to cache (same as gather_kv in paged path)
    k_full = np.concatenate([k_cache, k_new], axis=2)    # (1, n_head, ctx+1, head_dim)
    v_full_np = np.concatenate([v_cache, v_new_np], axis=2)

    k_full_t = tensor_from_numpy(k_full, backend=backend)
    v_full_t = tensor_from_numpy(v_full_np, backend=backend)
    kT_full = k_full_t.permute(0, 1, 3, 2)  # (1, n_head, head_dim, ctx+1)

    # For a single query token, the causal mask is trivially all-zeros (1x1),
    # so self_attention is correct here without modification.
    attn_out = attn.self_attention(q, kT_full, v_full_t)  # (1, 1, n_embd)
    output = attn.out_projection(
        attn_out.contiguous().view(1, n_embd)
    ).view(1, 1, n_embd)
    return output


# ============================================================================
# DATA VALIDATION
# ============================================================================

def check_data(n_embd=64, n_head=4, seq_len=32, block_size=8):
    """
    Sanity-check inputs and cache contents before running benchmarks.
    Catches shape mismatches, NaN/Inf, silent cache write failures, etc.
    """
    print(f"\n{'='*70}")
    print("DATA VALIDATION")
    print(f"  Config: n_embd={n_embd}, n_head={n_head}, seq_len={seq_len}, block_size={block_size}")
    print(f"{'='*70}")

    backend = get_backend()
    head_dim = n_embd // n_head

    np.random.seed(0)
    x_np = np.random.randn(1, seq_len, n_embd).astype(np.float32)
    x = tensor_from_numpy(x_np, backend=backend)

    attn = minitorch.MultiHeadAttention(
        n_embd=n_embd, n_head=n_head, causal=True,
        p_dropout=0.0, bias=False, backend=backend
    )

    # 1. Input sanity
    assert not np.isnan(x_np).any() and not np.isinf(x_np).any(), "Input contains NaN/Inf"
    print(f"  [OK] Input shape={x_np.shape}, range=[{x_np.min():.3f}, {x_np.max():.3f}]")

    # 2. Standard forward
    out_std = attn.forward(x, block_manager=None, layer_idx=None, seq_id=None).to_numpy()
    assert not np.isnan(out_std).any(), "Standard output contains NaN"
    assert out_std.shape == (1, seq_len, n_embd), f"Wrong shape: {out_std.shape}"
    print(f"  [OK] Standard output shape={out_std.shape}, range=[{out_std.min():.3f}, {out_std.max():.3f}]")

    # 3. Paged forward
    bm = BlockManager(num_layers=1, num_blocks=10, block_size=block_size,
                      n_head=n_head, head_dim=head_dim, backend=backend)
    bm.allocate_seq(seq_id=0)
    out_paged = attn.forward(x, block_manager=bm, layer_idx=0, seq_id=0).to_numpy()
    assert not np.isnan(out_paged).any(), "Paged output contains NaN"
    print(f"  [OK] Paged output shape={out_paged.shape}, range=[{out_paged.min():.3f}, {out_paged.max():.3f}]")

    # 4. KV cache was actually written
    kv_k_cpu = bm.kv_k
    assert kv_k_cpu[0].any(), "KV cache is all zeros after prefill — write_kv may be broken"
    print(f"  [OK] KV cache written: layer 0 has non-zero values")

    # 5. seq_lengths is NOT auto-updated by attn.forward (benchmark must set it manually)
    assert bm.seq_lengths[0] == 0, \
        f"seq_lengths was unexpectedly auto-updated to {bm.seq_lengths[0]}"
    print(f"  [OK] seq_lengths[0]={bm.seq_lengths[0]} (benchmark must set this manually after prefill)")

    # 6. Outputs match
    diff = np.abs(out_std - out_paged).max()
    assert diff < 1e-4, f"Standard and paged outputs diverge: max_diff={diff:.2e}"
    print(f"  [OK] Standard vs paged max_diff={diff:.2e}")

    # 7. Contiguous KV cache decode correctness
    x_token_np = np.random.randn(1, 1, n_embd).astype(np.float32)
    x_token = tensor_from_numpy(x_token_np, backend=backend)
    k_cache, v_cache = contiguous_prefill(attn, x)
    assert k_cache.shape == (1, n_head, seq_len, head_dim), \
        f"contiguous_prefill k_cache wrong shape: {k_cache.shape}"
    out_cont = contiguous_decode(attn, x_token, k_cache, v_cache, backend).to_numpy()
    assert not np.isnan(out_cont).any(), "Contiguous decode output contains NaN"
    assert out_cont.shape == (1, 1, n_embd), f"Wrong shape: {out_cont.shape}"
    print(f"  [OK] Contiguous decode shape={out_cont.shape}, range=[{out_cont.min():.3f}, {out_cont.max():.3f}]")

    # 8. Paged decode output matches contiguous decode output
    bm2 = BlockManager(num_layers=1, num_blocks=10, block_size=block_size,
                       n_head=n_head, head_dim=head_dim, backend=backend)
    bm2.allocate_seq(seq_id=0)
    _ = attn.forward(x, block_manager=bm2, layer_idx=0, seq_id=0)
    bm2.seq_lengths[0] = seq_len
    out_paged_dec = attn.forward(x_token, block_manager=bm2, layer_idx=0, seq_id=0).to_numpy()
    dec_diff = np.abs(out_cont - out_paged_dec).max()
    assert dec_diff < 1e-4, f"Contiguous vs paged decode diverge: max_diff={dec_diff:.2e}"
    print(f"  [OK] Contiguous vs paged decode max_diff={dec_diff:.2e}")

    print("\n  All checks passed!\n")
    return True


# ============================================================================
# CORRECTNESS TEST: Paged vs. Standard Attention (prefill)
# ============================================================================

def test_correctness(n_embd=256, n_head=8, seq_len=128, batch_size=1, block_size=16):
    """
    Verify paged and standard attention produce identical outputs on prefill.
    Note: both paths do the same O(T^2) work here, so timing is not meaningful.
    """
    print(f"\n{'='*70}")
    print(f"CORRECTNESS TEST")
    print(f"  Config: n_embd={n_embd}, n_head={n_head}, seq_len={seq_len}, batch={batch_size}")
    print(f"{'='*70}")

    backend = get_backend()

    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, n_embd).astype(np.float32)
    x_tensor = tensor_from_numpy(x_np, backend=backend, requires_grad=False)

    attn = minitorch.MultiHeadAttention(
        n_embd=n_embd, n_head=n_head, causal=True,
        p_dropout=0.0, bias=False, backend=backend
    )

    # Warmup both paths
    print("\n[Warmup] Running one pass each to warm up GPU...")
    _ = attn.forward(x_tensor, block_manager=None, layer_idx=None, seq_id=None)
    bm_warmup = BlockManager(
        num_layers=1,
        num_blocks=batch_size * ((seq_len // block_size) + 2),
        block_size=block_size, n_head=n_head, head_dim=n_embd // n_head, backend=backend,
    )
    bm_warmup.allocate_seq(seq_id=0)
    _ = attn.forward(x_tensor, block_manager=bm_warmup, layer_idx=0, seq_id=0)
    torch.cuda.synchronize()

    # Standard
    print("\n[1/2] Running standard attention (prefill)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_standard = attn.forward(x_tensor, block_manager=None, layer_idx=None, seq_id=None)
    torch.cuda.synchronize()
    time_standard = time.perf_counter() - t0
    print(f"      Time: {time_standard*1000:.3f} ms")

    # Paged
    print("\n[2/2] Running paged attention (prefill)...")
    bm = BlockManager(
        num_layers=1,
        num_blocks=batch_size * ((seq_len // block_size) + 2),
        block_size=block_size, n_head=n_head, head_dim=n_embd // n_head, backend=backend,
    )
    bm.allocate_seq(seq_id=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_paged = attn.forward(x_tensor, block_manager=bm, layer_idx=0, seq_id=0)
    torch.cuda.synchronize()
    time_paged = time.perf_counter() - t0
    print(f"      Time: {time_paged*1000:.3f} ms")

    out_std_np = output_standard.to_numpy()
    out_paged_np = output_paged.to_numpy()
    max_diff = np.abs(out_std_np - out_paged_np).max()
    mean_diff = np.abs(out_std_np - out_paged_np).mean()

    print(f"\n[RESULTS]")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Note: both paths do identical O(T^2) work during prefill,")
    print(f"        so timing difference here is noise, not a meaningful speedup.")

    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"  PASS: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"  FAIL: Outputs differ beyond tolerance ({tolerance})")
        return False


# ============================================================================
# BENCHMARK: LATENCY — Paged vs. Contiguous KV Cache (decode phase)
#
# Correct comparison:
#   Both paths prefill the same context into a KV cache.
#   On each decode trial, both compute Q/K/V for the new token and run
#   attention over the full cached history.
#   Difference: contiguous uses direct numpy indexing;
#               paged uses block table indirection.
# ============================================================================

def benchmark_latency(n_embd=256, n_head=8, batch_size=1, block_size=16, num_trials=10):
    print(f"\n{'='*80}")
    print(f"LATENCY BENCHMARK: Single-Token Decode")
    print(f"  Contiguous KV cache vs. Paged KV cache")
    print(f"  Config: n_embd={n_embd}, n_head={n_head}, batch={batch_size}, block_size={block_size}")
    print(f"  Trials: {num_trials} (reporting median +/- std)")
    print(f"{'='*80}")

    backend = get_backend()
    attn = minitorch.MultiHeadAttention(
        n_embd=n_embd, n_head=n_head, causal=True,
        p_dropout=0.0, bias=False, backend=backend
    )

    context_lengths = [256, 512, 1024, 2048, 4096]

    print(f"\n{'Context':<12} {'Contiguous (ms)':<22} {'Paged (ms)':<22} {'Ratio':<10}")
    print("-" * 80)

    for ctx_len in context_lengths:
        np.random.seed(42)

        x_context_np = np.random.randn(batch_size, ctx_len, n_embd).astype(np.float32)
        x_context = tensor_from_numpy(x_context_np, backend=backend)
        x_token_np = np.random.randn(batch_size, 1, n_embd).astype(np.float32)
        x_token = tensor_from_numpy(x_token_np, backend=backend)

        # ---- Setup contiguous KV cache ----
        k_cache, v_cache = contiguous_prefill(attn, x_context)

        # ---- Setup paged KV cache ----
        bm = BlockManager(
            num_layers=1,
            num_blocks=batch_size * ((ctx_len // block_size) + 2),
            block_size=block_size, n_head=n_head, head_dim=n_embd // n_head, backend=backend,
        )
        bm.allocate_seq(seq_id=0)
        _ = attn.forward(x_context, block_manager=bm, layer_idx=0, seq_id=0)
        bm.seq_lengths[0] = ctx_len

        # ---- Warmup both ----
        _ = contiguous_decode(attn, x_token, k_cache, v_cache, backend)
        torch.cuda.synchronize()
        _ = attn.forward(x_token, block_manager=bm, layer_idx=0, seq_id=0)
        torch.cuda.synchronize()

        # ---- Time contiguous decode ----
        times_cont = []
        torch.cuda.synchronize()
        for _ in range(num_trials):
            t0 = time.perf_counter()
            _ = contiguous_decode(attn, x_token, k_cache, v_cache, backend)
            torch.cuda.synchronize()
            times_cont.append((time.perf_counter() - t0) * 1000)

        # ---- Time paged decode ----
        times_paged = []
        torch.cuda.synchronize()
        for _ in range(num_trials):
            t0 = time.perf_counter()
            _ = attn.forward(x_token, block_manager=bm, layer_idx=0, seq_id=0)
            torch.cuda.synchronize()
            times_paged.append((time.perf_counter() - t0) * 1000)

        cont_median  = sorted(times_cont)[len(times_cont) // 2]
        paged_median = sorted(times_paged)[len(times_paged) // 2]
        cont_std     = stdev(times_cont)  if len(times_cont)  > 1 else 0
        paged_std    = stdev(times_paged) if len(times_paged) > 1 else 0
        ratio = paged_median / cont_median if cont_median > 0 else 0

        print(f"{ctx_len:<12} {cont_median:6.3f} +/- {cont_std:5.3f}       "
              f"{paged_median:6.3f} +/- {paged_std:5.3f}      {ratio:5.2f}x")

    print(f"\n  Ratio > 1.0 means paged is slower (block table overhead).")
    print(f"  Ratio ~1.0 means both caches have similar decode latency.")


# ============================================================================
# BENCHMARK: MEMORY — Multi-sequence, paged pool vs. per-sequence pre-alloc
#
# Correct scenario:
#   N sequences at random lengths between min_seq_len and max_seq_len.
#   Standard: each sequence pre-allocates a buffer of size max_seq_len
#             (worst-case reservation, typical in naive implementations).
#   Paged:    one shared block pool sized for the actual tokens in all seqs.
#   Fragmentation: standard wastes (max_seq_len - actual_len) per sequence.
#                  paged wastes at most (block_size - 1) per sequence.
# ============================================================================

def _measure_memory_one(n_layers, n_head, head_dim, block_size,
                         num_seqs, max_seq_len, min_seq_len, backend, seed=42):
    """
    Measure GPU memory for standard vs paged for one (num_seqs, utilization) scenario.
    Returns (mem_standard_mb, mem_paged_mb, theoretical_min_mb, seq_lengths).
    """
    np.random.seed(seed)
    seq_lengths = np.random.randint(min_seq_len, max_seq_len + 1, size=num_seqs)
    total_tokens = int(seq_lengths.sum())
    total_blocks = int(sum((int(l) + block_size - 1) // block_size for l in seq_lengths))

    # Standard
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mb0 = get_gpu_mem_mb()
    std_k = torch.zeros(num_seqs, n_layers, n_head, max_seq_len, head_dim,
                        dtype=torch.float32, device='cuda')
    std_v = torch.zeros(num_seqs, n_layers, n_head, max_seq_len, head_dim,
                        dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    mem_std = get_gpu_mem_mb() - mb0
    del std_k, std_v
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Paged
    mb0 = get_gpu_mem_mb()
    bm = BlockManager(num_layers=n_layers, num_blocks=total_blocks,
                      block_size=block_size, n_head=n_head, head_dim=head_dim,
                      backend=backend)
    torch.cuda.synchronize()
    mem_paged = get_gpu_mem_mb() - mb0
    del bm
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    theoretical_min = (total_tokens * n_layers * n_head * head_dim * 2 * 4) / 1024 / 1024
    return mem_std, mem_paged, theoretical_min, seq_lengths


def benchmark_memory(n_embd=1024, n_head=16, block_size=16, n_layers=12,
                     num_seqs=16, max_seq_len=2048):
    """
    Compare GPU memory: standard (pre-alloc max_seq_len) vs paged (actual tokens).

    Uses realistic model parameters (n_embd=1024, n_layers=12 ~ GPT-2 medium)
    and sweeps utilization ratios to show how savings scale.

    Utilization = avg_actual_len / max_seq_len.
    Low utilization (short actual sequences against a high max) → high savings.
    """
    print(f"\n{'='*70}")
    print(f"MEMORY BENCHMARK (GPU Memory, Multi-Sequence)")
    print(f"  Model config: n_embd={n_embd}, n_head={n_head}, n_layers={n_layers}")
    print(f"  Sequences: {num_seqs}, max_seq_len={max_seq_len}, block_size={block_size}")
    print(f"  (Sweeping utilization = avg_actual_len / max_seq_len)")
    print(f"{'='*70}")

    head_dim = n_embd // n_head
    backend = get_backend()

    # ---- Single-scenario result (the main number to report) ----
    # Use min=256, showing realistic variance: short prompts against a 2048 max
    mem_std, mem_paged, theoretical_min, seq_lengths = _measure_memory_one(
        n_layers, n_head, head_dim, block_size,
        num_seqs, max_seq_len, min_seq_len=256, backend=backend
    )

    avg_len = int(seq_lengths.mean())
    utilization = avg_len / max_seq_len * 100

    print(f"\n  Sequence lengths: {seq_lengths.tolist()}")
    print(f"  Avg actual length: {avg_len}  ({utilization:.1f}% utilization of max={max_seq_len})")

    if mem_std > 0 and mem_paged > 0:
        savings = (mem_std - mem_paged) / mem_std * 100
        waste_std   = (mem_std   - theoretical_min) / mem_std   * 100
        waste_paged = (mem_paged - theoretical_min) / mem_paged * 100
        print(f"\n  Standard (pre-alloc {max_seq_len} per seq): {mem_std:7.2f} MB  "
              f"({waste_std:.1f}% wasted)")
        print(f"  Paged    (actual blocks only):       {mem_paged:7.2f} MB  "
              f"({waste_paged:.1f}% wasted, at most {block_size-1} slots/seq)")
        print(f"  Theoretical minimum:                 {theoretical_min:7.2f} MB")
        print(f"\n  Memory savings: {savings:.1f}%  "
              f"({mem_std:.2f} MB -> {mem_paged:.2f} MB)")
    else:
        print(f"\n  WARNING: pynvml not available — install nvidia-ml-py.")
        return

    # ---- Utilization sweep: show savings scale as sequences get shorter ----
    print(f"\n  Savings vs utilization (how full sequences are relative to max_seq_len):")
    print(f"  {'Utilization':<14} {'Avg len':<10} {'Standard MB':<14} "
          f"{'Paged MB':<12} {'Savings':<10}")
    print(f"  {'-'*60}")

    # Sweep: min_seq_len from 10% to 90% of max_seq_len
    for pct in [10, 25, 50, 75, 90]:
        min_len = max(1, int(max_seq_len * pct / 100) - 100)
        max_len = max(min_len + 1, int(max_seq_len * pct / 100) + 100)
        max_len = min(max_len, max_seq_len)

        ms, mp, _, sl = _measure_memory_one(
            n_layers, n_head, head_dim, block_size,
            num_seqs, max_seq_len, min_seq_len=min_len, backend=backend, seed=pct
        )
        avg = int(sl.mean())
        util = avg / max_seq_len * 100
        if ms > 0 and mp > 0:
            sav = (ms - mp) / ms * 100
            print(f"  {util:5.1f}%         {avg:<10} {ms:<14.2f} {mp:<12.2f} {sav:.1f}%")
        else:
            print(f"  {pct}%  (pynvml unavailable)")

    print(f"\n  Key insight: savings increase as sequences are shorter relative to max_seq_len.")
    print(f"  Standard always pays for {max_seq_len} slots regardless of actual use.")

    # ---- Per-sequence fragmentation distribution ----
    # Shows the spread, not just the mean.
    head_dim = n_embd // n_head
    wasted_std_per_seq = np.array([max_seq_len - int(l) for l in seq_lengths])
    # Paged: last block of each seq may be partially filled
    wasted_paged_per_seq = np.array([
        (((int(l) - 1) // block_size) + 1) * block_size - int(l)
        for l in seq_lengths
    ])

    print(f"\n  Per-sequence fragmentation (wasted slots):")
    print(f"  {'':20} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    print(f"  {'Standard':20} {wasted_std_per_seq.mean():8.1f} "
          f"{wasted_std_per_seq.std():8.1f} "
          f"{wasted_std_per_seq.min():6d} {wasted_std_per_seq.max():6d}")
    print(f"  {'Paged':20} {wasted_paged_per_seq.mean():8.1f} "
          f"{wasted_paged_per_seq.std():8.1f} "
          f"{wasted_paged_per_seq.min():6d} {wasted_paged_per_seq.max():6d}")
    print(f"  (Paged max waste is always block_size-1 = {block_size-1} slots)")


# ============================================================================
# BENCHMARK: OOM DEMONSTRATION
#
# Directly shows the capacity gap: pick a batch size that exceeds standard's
# memory budget and confirm standard raises an OOM while paged succeeds.
# ============================================================================

def benchmark_oom(n_embd=1024, n_head=16, block_size=16, n_layers=12,
                  actual_seq_len=512, max_seq_len=2048):
    """
    Demonstrate that a batch size standard cannot fit in GPU memory is handled
    fine by the paged pool.

    Standard pre-allocates max_seq_len per sequence; paged allocates only the
    blocks for actual_seq_len.  We pick a batch size just above what standard
    can hold and attempt both allocations, catching the OOM for standard.
    """
    print(f"\n{'='*70}")
    print(f"OOM DEMONSTRATION")
    print(f"  Model: n_embd={n_embd}, n_head={n_head}, n_layers={n_layers}")
    print(f"  actual_seq_len={actual_seq_len}, max_seq_len={max_seq_len}")
    print(f"{'='*70}")

    head_dim = n_embd // n_head
    backend = get_backend()

    bytes_per_seq_std   = n_layers * n_head * max_seq_len  * head_dim * 2 * 4
    blocks_per_seq = (actual_seq_len + block_size - 1) // block_size
    bytes_per_seq_paged = n_layers * blocks_per_seq * n_head * block_size * head_dim * 2 * 4
    mb_per_seq_std   = bytes_per_seq_std   / 1024 / 1024
    mb_per_seq_paged = bytes_per_seq_paged / 1024 / 1024

    free_mb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated()) / 1024 / 1024

    max_std   = max(1, int(free_mb // mb_per_seq_std))
    max_paged = max(1, int(free_mb // mb_per_seq_paged))

    # Target: 20% above what standard can fit (but within paged's budget)
    target_batch = int(max_std * 1.2)
    target_batch = min(target_batch, max_paged)

    print(f"\n  Available GPU memory:  {free_mb:.0f} MB")
    print(f"  Per-seq cost standard: {mb_per_seq_std:.2f} MB  "
          f"(pre-alloc {max_seq_len} slots)")
    print(f"  Per-seq cost paged:    {mb_per_seq_paged:.2f} MB  "
          f"(actual {actual_seq_len} tokens)")
    print(f"  Max batch standard:    {max_std}")
    print(f"  Max batch paged:       {max_paged}")
    print(f"\n  Testing batch_size={target_batch} (20% above standard's limit)...")

    # Standard attempt
    torch.cuda.empty_cache()
    try:
        std_k = torch.zeros(target_batch, n_layers, n_head, max_seq_len, head_dim,
                            dtype=torch.float32, device='cuda')
        std_v = torch.zeros(target_batch, n_layers, n_head, max_seq_len, head_dim,
                            dtype=torch.float32, device='cuda')
        torch.cuda.synchronize()
        del std_k, std_v
        torch.cuda.empty_cache()
        print(f"  Standard: OK (unexpected — GPU may have more free memory than estimated)")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  Standard: OOM  — cannot allocate {target_batch * mb_per_seq_std:.0f} MB "
              f"for {target_batch} sequences")

    # Paged attempt
    torch.cuda.empty_cache()
    total_blocks = target_batch * (blocks_per_seq + 1)
    try:
        bm = BlockManager(num_layers=n_layers, num_blocks=total_blocks,
                          block_size=block_size, n_head=n_head, head_dim=head_dim,
                          backend=backend)
        torch.cuda.synchronize()
        paged_mb = total_blocks * n_layers * n_head * block_size * head_dim * 2 * 4 / 1024 / 1024
        del bm
        torch.cuda.empty_cache()
        print(f"  Paged:    OK  — allocated {paged_mb:.0f} MB for {target_batch} sequences "
              f"({actual_seq_len} actual tokens each)")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  Paged:    OOM  — target batch too large even for paged pool")


# ============================================================================
# BENCHMARK: THROUGHPUT — tokens/second at increasing batch sizes
#
# This is the primary metric from the vLLM paper.
# Paged attention uses less memory per sequence -> can fit more sequences
# concurrently -> larger effective batch -> more tokens/second.
#
# Standard approach: pre-allocate max_seq_len per sequence in a contiguous
#   buffer.  At some batch size this exhausts GPU memory (OOM).
# Paged approach: shared block pool sized to actual tokens.  Fits larger
#   batches in the same GPU budget.
#
# We simulate this without actually OOMing by:
#   1. Measuring how much GPU memory each approach needs per sequence.
#   2. Computing the max batch size each can fit in a fixed GPU budget.
#   3. Running that batch size and measuring tokens/second.
# ============================================================================

def benchmark_throughput(n_embd=256, n_head=8, block_size=16,
                         actual_seq_len=512, max_seq_len=2048, decode_steps=20,
                         gpu_budget_mb=4000, n_layers=4, num_trials=3, max_batch=32):
    """
    Compare tokens/second at the maximum batch size each approach can support
    within a fixed GPU memory budget.

    The key distinction:
      Standard pre-allocates max_seq_len per sequence (naive implementation —
        must reserve worst-case upfront since buffer is contiguous).
      Paged allocates only the blocks actually needed for actual_seq_len.

    This gap (max_seq_len vs actual_seq_len) is what drives memory savings and
    the higher batch size for paged.

    Args:
        actual_seq_len: Actual prompt length (tokens in use).
        max_seq_len:    Standard pre-allocates this many slots per sequence.
        gpu_budget_mb:  GPU memory budget to simulate (MB).
        decode_steps:   Number of decode steps per trial.
        n_layers:       Number of transformer layers in the KV cache.
    """
    print(f"\n{'='*80}")
    print(f"THROUGHPUT BENCHMARK: tokens/second at max batch size")
    print(f"  Config: n_embd={n_embd}, n_head={n_head}, block_size={block_size}")
    print(f"  Actual seq len: {actual_seq_len}  |  Standard pre-alloc: {max_seq_len}")
    print(f"  GPU budget: {gpu_budget_mb} MB  |  decode_steps={decode_steps}  |  layers={n_layers}")
    print(f"{'='*80}")

    head_dim = n_embd // n_head
    backend = get_backend()

    # ---- Memory cost per sequence ----
    # Standard: must pre-allocate max_seq_len (can't grow a contiguous buffer)
    bytes_per_seq_standard = n_layers * n_head * max_seq_len * head_dim * 2 * 4  # float32
    mb_per_seq_standard = bytes_per_seq_standard / 1024 / 1024

    # Paged: allocates only blocks for actual tokens used
    blocks_per_seq = (actual_seq_len + block_size - 1) // block_size
    bytes_per_seq_paged = n_layers * blocks_per_seq * n_head * block_size * head_dim * 2 * 4
    mb_per_seq_paged = bytes_per_seq_paged / 1024 / 1024

    max_batch_standard = max(1, int(gpu_budget_mb // mb_per_seq_standard))
    max_batch_paged    = max(1, int(gpu_budget_mb // mb_per_seq_paged))

    print(f"\n  Memory per sequence:")
    print(f"    Standard (contiguous, pre-alloc={max_seq_len}): {mb_per_seq_standard:.2f} MB")
    print(f"    Paged    (block pool, {blocks_per_seq} blocks for {actual_seq_len} tokens): {mb_per_seq_paged:.2f} MB")
    print(f"\n  Max batch size within {gpu_budget_mb} MB budget:")
    print(f"    Standard: {max_batch_standard} sequences")
    print(f"    Paged:    {max_batch_paged} sequences")

    attn = minitorch.MultiHeadAttention(
        n_embd=n_embd, n_head=n_head, causal=True,
        p_dropout=0.0, bias=False, backend=backend
    )

    def run_standard_batch(batch_size):
        """Prefill + decode_steps decode steps using contiguous KV cache per sequence."""
        np.random.seed(42)
        x_ctx_np = np.random.randn(1, actual_seq_len, n_embd).astype(np.float32)
        x_ctx = tensor_from_numpy(x_ctx_np, backend=backend)
        x_tok_np = np.random.randn(1, 1, n_embd).astype(np.float32)
        x_tok = tensor_from_numpy(x_tok_np, backend=backend)

        # Prefill all sequences (each independently — standard has no sharing)
        caches = []
        for _ in range(batch_size):
            k, v = contiguous_prefill(attn, x_ctx)
            caches.append((k, v))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_tokens = 0
        for _ in range(decode_steps):
            for k_cache, v_cache in caches:
                contiguous_decode(attn, x_tok, k_cache, v_cache, backend)
                total_tokens += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return total_tokens / elapsed

    def run_paged_batch(batch_size):
        """Prefill + decode_steps decode steps using a shared paged KV cache."""
        np.random.seed(42)
        x_ctx_np = np.random.randn(1, actual_seq_len, n_embd).astype(np.float32)
        x_ctx = tensor_from_numpy(x_ctx_np, backend=backend)
        x_tok_np = np.random.randn(1, 1, n_embd).astype(np.float32)
        x_tok = tensor_from_numpy(x_tok_np, backend=backend)

        total_blocks = batch_size * (blocks_per_seq + 2)
        bm = BlockManager(
            num_layers=n_layers,
            num_blocks=total_blocks,
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
            backend=backend,
        )

        # Prefill all sequences into the shared pool
        for seq_id in range(batch_size):
            bm.allocate_seq(seq_id)
            _ = attn.forward(x_ctx, block_manager=bm, layer_idx=0, seq_id=seq_id)
            bm.seq_lengths[seq_id] = actual_seq_len

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_tokens = 0
        for _ in range(decode_steps):
            for seq_id in range(batch_size):
                attn.forward(x_tok, block_manager=bm, layer_idx=0, seq_id=seq_id)
                total_tokens += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return total_tokens / elapsed

    # ---- Warmup ----
    print(f"\n  [Warmup]...")
    run_standard_batch(1)
    run_paged_batch(1)

    # ---- Sweep batch sizes up to max for each approach, capped at max_batch ----
    eff_std   = min(max_batch_standard, max_batch)
    eff_paged = min(max_batch_paged,    max_batch)

    all_batch_sizes = sorted(set(
        [1, 2, 4, 8, 16] + [eff_std, eff_paged]
    ))
    all_batch_sizes = [b for b in all_batch_sizes if b >= 1]

    print(f"\n  {'Batch':<8} {'Standard (tok/s)':<22} {'Paged (tok/s)':<22} {'Speedup':<10} {'Note'}")
    print(f"  {'-'*75}")

    for bs in all_batch_sizes:
        # Standard
        if bs <= max_batch_standard:
            tps_std_trials = [run_standard_batch(bs) for _ in range(num_trials)]
            tps_std = sorted(tps_std_trials)[num_trials // 2]
            std_str = f"{tps_std:8.1f}"
        else:
            tps_std = None
            std_str = "     OOM"

        # Paged
        if bs <= max_batch_paged:
            tps_paged_trials = [run_paged_batch(bs) for _ in range(num_trials)]
            tps_paged = sorted(tps_paged_trials)[num_trials // 2]
            paged_str = f"{tps_paged:8.1f}"
        else:
            tps_paged = None
            paged_str = "     OOM"

        if tps_std is not None and tps_paged is not None:
            speedup = tps_paged / tps_std
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "   -"

        note = ""
        if bs > max_batch_standard and bs <= max_batch_paged:
            note = "<-- paged only (standard would OOM)"

        print(f"  {bs:<8} {std_str:<22} {paged_str:<22} {speedup_str:<10} {note}")

    print(f"\n  At batch={max_batch_paged} paged fits {max_batch_paged/max(max_batch_standard,1):.1f}x "
          f"more sequences than standard in the same GPU budget.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark paged attention")
    parser.add_argument("--check-data",        action="store_true", help="Run data validation checks")
    parser.add_argument("--test-correctness",  action="store_true")
    parser.add_argument("--bench-latency",     action="store_true")
    parser.add_argument("--bench-memory",      action="store_true")
    parser.add_argument("--bench-throughput",  action="store_true")
    parser.add_argument("--bench-oom",         action="store_true", help="OOM demonstration")
    parser.add_argument("--all",               action="store_true", help="Run all benchmarks")
    parser.add_argument("--gpu-budget-mb",     type=int, default=4000,
                        help="GPU memory budget in MB for throughput benchmark (default: 4000)")

    args = parser.parse_args()

    if args.all:
        check_data()
        test_correctness(n_embd=256, n_head=8, seq_len=128, batch_size=1, block_size=16)
        benchmark_latency(n_embd=256, n_head=8, batch_size=1, block_size=16, num_trials=10)
        benchmark_memory(n_embd=1024, n_head=16, block_size=16,
                         n_layers=12, num_seqs=16, max_seq_len=2048)
        benchmark_oom(n_embd=1024, n_head=16, block_size=16, n_layers=12,
                      actual_seq_len=512, max_seq_len=2048)
        benchmark_throughput(n_embd=256, n_head=8, block_size=16,
                             actual_seq_len=512, max_seq_len=2048, decode_steps=20,
                             gpu_budget_mb=args.gpu_budget_mb, n_layers=4, num_trials=3,
                             max_batch=32)
    else:
        if args.check_data:
            check_data()
        if args.test_correctness:
            test_correctness(n_embd=256, n_head=8, seq_len=128, batch_size=1, block_size=16)
        if args.bench_latency:
            benchmark_latency(n_embd=256, n_head=8, batch_size=1, block_size=16, num_trials=10)
        if args.bench_memory:
            benchmark_memory(n_embd=1024, n_head=16, block_size=16,
                             n_layers=12, num_seqs=16, max_seq_len=2048)
        if args.bench_throughput:
            benchmark_throughput(n_embd=256, n_head=8, block_size=16,
                                 actual_seq_len=512, max_seq_len=2048, decode_steps=20,
                                 gpu_budget_mb=args.gpu_budget_mb, n_layers=4, num_trials=3)
        if args.bench_oom:
            benchmark_oom(n_embd=1024, n_head=16, block_size=16, n_layers=12,
                          actual_seq_len=512, max_seq_len=2048)
