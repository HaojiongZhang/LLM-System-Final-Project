import time
import copy
import numpy as np
import sys
sys.path.append("./")
import minitorch
from minitorch import DecoderLM
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.cuda_kernel_ops import CudaKernelOps


def make_backend(use_cpu: bool = False):
    if use_cpu:
        print("Using CPU backend")
        return minitorch.TensorBackend()
    print("Using CUDA backend")
    return minitorch.TensorBackend(CudaKernelOps)


def sync_if_needed():
    # Your ctypes CUDA wrapper already does syncs internally,
    # so this is mostly a no-op placeholder for readability.
    pass


def clone_parameters(dst_model, src_model):
    """
    Copy parameter values from src_model to dst_model in-place.
    Assumes same architecture.
    """
    dst_params = list(dst_model.parameters())
    src_params = list(src_model.parameters())
    assert len(dst_params) == len(src_params), "Parameter count mismatch"

    for dp, sp in zip(dst_params, src_params):
        assert dp.value.shape == sp.value.shape, f"Shape mismatch: {dp.value.shape} vs {sp.value.shape}"
        dp.update(tensor_from_numpy(sp.value.to_numpy().copy(), backend=dp.value.backend))


def max_abs_diff(a, b):
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    return float(np.max(np.abs(a_np - b_np)))


def mean_abs_diff(a, b):
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    return float(np.mean(np.abs(a_np - b_np)))


def run_once(model, input_ids):
    sync_if_needed()
    t0 = time.time()
    out = model(input_ids)
    sync_if_needed()
    t1 = time.time()
    return out, (t1 - t0)


def benchmark_model(model, batches, warmup=2, iters=5):
    """
    Returns avg latency in seconds over all timed iterations.
    """
    # warmup
    for i in range(min(warmup, len(batches))):
        _ = model(batches[i])

    sync_if_needed()

    times = []
    for _ in range(iters):
        for batch in batches:
            _, dt = run_once(model, batch)
            times.append(dt)

    return float(np.mean(times)), float(np.std(times))


def make_random_batches(
    backend,
    n_vocab,
    batch_size,
    seq_len,
    num_batches,
    seed=0,
):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_batches):
        x = rng.integers(
            low=0,
            high=n_vocab,
            size=(batch_size, seq_len),
            dtype=np.int32,
        )
        batches.append(tensor_from_numpy(x, backend=backend))
    return batches


def main():
    # Small-ish config first; scale up after correctness looks good.
    use_cpu = False
    seed = 1234
    np.random.seed(seed)

    backend = make_backend(use_cpu=use_cpu)

    config = {
        "n_vocab": 5000,
        "n_embd": 256,
        "n_head": 8,
        "n_positions": 128,
        "p_dropout": 0.0,   # IMPORTANT: inference comparison should disable dropout
        "ln_eps": 1e-5,
        "bias": True,
        "backend": backend,
    }

    baseline_model = DecoderLM(**config, attention_impl="baseline")
    flash_model = DecoderLM(**config, attention_impl="flash")

    # Make sure both models have identical weights
    clone_parameters(flash_model, baseline_model)

    # Inference-only benchmark
    baseline_model.eval()
    flash_model.eval()

    # Try a few shapes
    shapes = [
        (4, 32),
        (4, 64),
        (8, 64),
        (8, 96),
    ]

    for batch_size, seq_len in shapes:
        print("=" * 80)
        print(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")

        batches = make_random_batches(
            backend=backend,
            n_vocab=config["n_vocab"],
            batch_size=batch_size,
            seq_len=seq_len,
            num_batches=3,
            seed=seed + batch_size + seq_len,
        )

        # correctness check on one batch
        baseline_out, baseline_dt = run_once(baseline_model, batches[0])
        flash_out, flash_dt = run_once(flash_model, batches[0])

        mad = mean_abs_diff(baseline_out, flash_out)
        maxd = max_abs_diff(baseline_out, flash_out)

        print(f"Single-run baseline latency: {baseline_dt:.6f}s")
        print(f"Single-run flash    latency: {flash_dt:.6f}s")
        print(f"Mean abs diff: {mad:.8f}")
        print(f"Max  abs diff: {maxd:.8f}")

        # timed benchmark
        base_mean, base_std = benchmark_model(baseline_model, batches, warmup=2, iters=5)
        flash_mean, flash_std = benchmark_model(flash_model, batches, warmup=2, iters=5)

        tokens_per_batch = batch_size * seq_len
        base_tps = tokens_per_batch / base_mean
        flash_tps = tokens_per_batch / flash_mean
        speedup = base_mean / flash_mean

        print(f"Baseline avg latency: {base_mean:.6f}s ± {base_std:.6f}")
        print(f"Flash    avg latency: {flash_mean:.6f}s ± {flash_std:.6f}")
        print(f"Baseline tokens/s:   {base_tps:.2f}")
        print(f"Flash    tokens/s:   {flash_tps:.2f}")
        print(f"Speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()