# import time
# import numpy as np
# import sys
# sys.path.append("./")
# import minitorch

# from minitorch.cuda_kernel_ops import CudaKernelOps
# from minitorch.tensor_functions import tensor_from_numpy
# from minitorch.transformer import MultiHeadAttention  # adjust if your import path differs

# def cuda_sync():
#     CudaKernelOps.synchronize()
# def make_backend():
#     return minitorch.TensorBackend(CudaKernelOps)


# def clone_parameters(dst_module, src_module, backend):
#     dst_params = list(dst_module.parameters())
#     src_params = list(src_module.parameters())
#     assert len(dst_params) == len(src_params), "Parameter count mismatch"

#     for dp, sp in zip(dst_params, src_params):
#         arr = sp.value.to_numpy().copy()
#         dp.update(tensor_from_numpy(arr, backend=backend))


# def benchmark_call(fn, warmup=10, iters=50):
#     for _ in range(warmup):
#         _ = fn()
#     cuda_sync()

#     times = []
#     out = None
#     for _ in range(iters):
#         cuda_sync()
#         t0 = time.perf_counter()
#         out = fn()
#         cuda_sync()
#         t1 = time.perf_counter()
#         times.append(t1 - t0)

#     times = np.array(times, dtype=np.float64)
#     return out, float(times.mean()), float(times.std())

# def summarize_diff(a, b):
#     a_np = a.to_numpy()
#     b_np = b.to_numpy()
#     abs_diff = np.abs(a_np - b_np)
#     return float(abs_diff.mean()), float(abs_diff.max())


# def run_case(batch_size, seq_len, n_embd, n_head, causal=True):
#     backend = make_backend()

#     baseline = MultiHeadAttention(
#         n_embd=n_embd,
#         n_head=n_head,
#         causal=causal,
#         p_dropout=0.0,
#         backend=backend,
#         attention_impl="baseline",
#     )
#     flash = MultiHeadAttention(
#         n_embd=n_embd,
#         n_head=n_head,
#         causal=causal,
#         p_dropout=0.0,
#         backend=backend,
#         attention_impl="flash",
#     )

#     # Ensure identical projections / output weights
#     clone_parameters(flash, baseline, backend)

#     x_np = np.random.randn(batch_size, seq_len, n_embd).astype(np.float32)
#     x = tensor_from_numpy(x_np, backend=backend)

#     baseline.eval()
#     flash.eval()

#     # Project once so we only benchmark self_attention
#     q_base, kT_base, v_base = baseline.project_to_query_key_value(x)
#     q_flash, kT_flash, v_flash = flash.project_to_query_key_value(x)

#     # Numerical sanity: because weights are the same, projected tensors should match too
#     q_mean, q_max = summarize_diff(q_base, q_flash)
#     k_mean, k_max = summarize_diff(kT_base, kT_flash)
#     v_mean, v_max = summarize_diff(v_base, v_flash)

#     print(f"\nProjected tensor diffs:")
#     print(f"  q  mean={q_mean:.8e}, max={q_max:.8e}")
#     print(f"  kT mean={k_mean:.8e}, max={k_max:.8e}")
#     print(f"  v  mean={v_mean:.8e}, max={v_max:.8e}")

#     # Benchmark only self_attention
#     out_base, base_mean, base_std = benchmark_call(
#         lambda: baseline.self_attention(q_base, kT_base, v_base),
#         warmup=3,
#         iters=10,
#     )
#     out_flash, flash_mean, flash_std = benchmark_call(
#         lambda: flash.self_attention(q_flash, kT_flash, v_flash),
#         warmup=3,
#         iters=10,
#     )

#     mean_diff, max_diff = summarize_diff(out_base, out_flash)
#     tokens = batch_size * seq_len
#     base_tps = tokens / base_mean
#     flash_tps = tokens / flash_mean
#     speedup = base_mean / flash_mean

#     print("=" * 90)
#     print(
#         f"batch={batch_size:>3} seq={seq_len:>4} embd={n_embd:>4} "
#         f"heads={n_head:>2} head_dim={n_embd // n_head:>3} causal={causal}"
#     )
#     print(f"baseline self_attention: {base_mean:.6f}s ± {base_std:.6f}")
#     print(f"flash    self_attention: {flash_mean:.6f}s ± {flash_std:.6f}")
#     print(f"baseline tok/s: {base_tps:.2f}")
#     print(f"flash    tok/s: {flash_tps:.2f}")
#     print(f"speedup: {speedup:.3f}x")
#     print(f"output mean abs diff: {mean_diff:.8e}")
#     print(f"output max  abs diff: {max_diff:.8e}")

#     return {
#         "batch_size": batch_size,
#         "seq_len": seq_len,
#         "n_embd": n_embd,
#         "n_head": n_head,
#         "head_dim": n_embd // n_head,
#         "causal": causal,
#         "baseline_mean_s": base_mean,
#         "flash_mean_s": flash_mean,
#         "speedup": speedup,
#         "baseline_tok_s": base_tps,
#         "flash_tok_s": flash_tps,
#         "mean_abs_diff": mean_diff,
#         "max_abs_diff": max_diff,
#     }


# def main():
#     np.random.seed(0)

#     # Start with cases where flash should have more chance to help.
#     cases = [
#         (8, 64, 256, 8),
#         (8, 128, 256, 8),
#         (8, 256, 256, 8),
#         (8, 128, 512, 8),
#         (8, 256, 512, 8),
#         (4, 512, 512, 8),
#     ]

#     results = []
#     for batch_size, seq_len, n_embd, n_head in cases:
#         results.append(run_case(batch_size, seq_len, n_embd, n_head, causal=True))

#     print("\n" + "#" * 90)
#     print("Summary")
#     print("#" * 90)
#     for r in results:
#         print(
#             f"bs={r['batch_size']:>3} seq={r['seq_len']:>4} "
#             f"embd={r['n_embd']:>4} hd={r['head_dim']:>3} "
#             f"speedup={r['speedup']:.3f}x "
#             f"base={r['baseline_mean_s']:.6f}s "
#             f"flash={r['flash_mean_s']:.6f}s "
#             f"mean_diff={r['mean_abs_diff']:.3e} "
#             f"max_diff={r['max_abs_diff']:.3e}"
#         )


# if __name__ == "__main__":
#     main()


import argparse
import time
import numpy as np
import sys
sys.path.append("./")
import minitorch

from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.transformer import MultiHeadAttention


def cuda_sync():
    CudaKernelOps.synchronize()


def make_backend():
    return minitorch.TensorBackend(CudaKernelOps)


def benchmark_call(fn, warmup=10, iters=20):
    for _ in range(warmup):
        _ = fn()
    cuda_sync()

    times = []
    out = None
    for _ in range(iters):
        cuda_sync()
        t0 = time.perf_counter()
        out = fn()
        cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    return out, float(times.mean()), float(times.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["baseline", "flash"], required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--embd", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(0)
    backend = make_backend()

    mha = MultiHeadAttention(
        n_embd=args.embd,
        n_head=args.heads,
        causal=True,
        p_dropout=0.0,
        backend=backend,
        attention_impl=args.impl,
    )
    mha.eval()

    x_np = np.random.randn(args.bs, args.seq, args.embd).astype(np.float32)
    x = tensor_from_numpy(x_np, backend=backend)

    q, kT, v = mha.project_to_query_key_value(x)

    out, mean_t, std_t = benchmark_call(
        lambda: mha.self_attention(q, kT, v),
        warmup=args.warmup,
        iters=args.iters,
    )

    print(f"impl={args.impl}")
    print(f"bs={args.bs} seq={args.seq} embd={args.embd} heads={args.heads}")
    print(f"time={mean_t:.6f}s ± {std_t:.6f}")


if __name__ == "__main__":
    main()
    
# nsys profile   --output flash_seq256   --force-overwrite true   --trace=cuda,nvtx,osrt   --sample=none   python tests/test_flashforward.py --impl flash --bs 8 --seq 256 --embd 256 --heads 8