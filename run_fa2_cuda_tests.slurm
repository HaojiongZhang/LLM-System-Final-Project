#!/bin/bash
#SBATCH --job-name=fa2-cuda-test
#SBATCH -p GPU-shared
#SBATCH --gpus=v100:1
#SBATCH -t 00:20:00
#SBATCH --mem=16G
#SBATCH -o /jet/home/kchien/LLM-System-Final-Project-/fa2_cuda_test_%j.log

set -euo pipefail

cd /jet/home/kchien/LLM-System-Final-Project-

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"

module load cuda
source /jet/home/kchien/llmsys_hw3/.venv/bin/activate

# Build CUDA shared library with FA2 backward symbol.
bash ./compile_cuda.sh

# Quick symbol check.
nm -D minitorch/cuda_kernels/combine.so | grep FlashAttention2Backward

# Run existing FA2 backward tests (reference path + API checks).
pytest -q tests/test_flash_attention2_backward.py

# Explicitly run CUDA kernel path once for runtime validation.
python - <<'PY'
import numpy as np
import time
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.flash_attention2 import flash_attention2_backward

rng = np.random.default_rng(123)
B, H, T, D = 1, 1, 8, 8
scale = 1.0 / np.sqrt(float(D))

q = rng.normal(size=(B, H, T, D)).astype(np.float32)
k = rng.normal(size=(B, H, T, D)).astype(np.float32)
v = rng.normal(size=(B, H, T, D)).astype(np.float32)
dout = rng.normal(size=(B, H, T, D)).astype(np.float32)

scores = (q[0,0] @ k[0,0].T) * scale
m = np.max(scores, axis=-1, keepdims=True)
ex = np.exp(scores - m)
z = np.sum(ex, axis=-1, keepdims=True)
p = ex / z
out = np.zeros_like(q)
out[0,0] = p @ v[0,0]
lse = np.zeros((B, H, T), dtype=np.float32)
lse[0,0] = (np.log(z) + m)[:, 0]

dq, dk, dv = flash_attention2_backward(
    dout=tensor_from_numpy(dout),
    q=tensor_from_numpy(q),
    k=tensor_from_numpy(k),
    v=tensor_from_numpy(v),
    out=tensor_from_numpy(out),
    logsumexp=tensor_from_numpy(lse),
    causal=False,
    softmax_scale=scale,
    use_cuda_kernel=True,
)

print("[CUDA CHECK] success")
print("[CUDA CHECK] dq mean:", float(np.mean(dq.to_numpy())))
print("[CUDA CHECK] dk mean:", float(np.mean(dk.to_numpy())))
print("[CUDA CHECK] dv mean:", float(np.mean(dv.to_numpy())))

# Lightweight timing comparison (CUDA path vs NumPy blocked reference path).
# This is a practical sanity check, not a strict benchmark.
B, H, T, D = 2, 4, 128, 64
scale = 1.0 / np.sqrt(float(D))
q = rng.normal(size=(B, H, T, D)).astype(np.float32)
k = rng.normal(size=(B, H, T, D)).astype(np.float32)
v = rng.normal(size=(B, H, T, D)).astype(np.float32)
dout = rng.normal(size=(B, H, T, D)).astype(np.float32)

scores = (q[0, 0] @ k[0, 0].T) * scale
m = np.max(scores, axis=-1, keepdims=True)
ex = np.exp(scores - m)
z = np.sum(ex, axis=-1, keepdims=True)

# Build full out/lse for all heads/batches.
out = np.zeros_like(q)
lse = np.zeros((B, H, T), dtype=np.float32)
for b in range(B):
    for h in range(H):
        s = (q[b, h] @ k[b, h].T) * scale
        mm = np.max(s, axis=-1, keepdims=True)
        ee = np.exp(s - mm)
        zz = np.sum(ee, axis=-1, keepdims=True)
        pp = ee / zz
        out[b, h] = pp @ v[b, h]
        lse[b, h] = (np.log(zz) + mm)[:, 0]

qd = tensor_from_numpy(q)
kd = tensor_from_numpy(k)
vd = tensor_from_numpy(v)
dod = tensor_from_numpy(dout)
od = tensor_from_numpy(out)
lsed = tensor_from_numpy(lse)

# Warmup CUDA path once.
flash_attention2_backward(
    dout=dod, q=qd, k=kd, v=vd, out=od, logsumexp=lsed,
    causal=False, softmax_scale=scale, use_cuda_kernel=True,
)

t0 = time.perf_counter()
flash_attention2_backward(
    dout=dod, q=qd, k=kd, v=vd, out=od, logsumexp=lsed,
    causal=False, softmax_scale=scale, use_cuda_kernel=True,
)
t1 = time.perf_counter()

t2 = time.perf_counter()
flash_attention2_backward(
    dout=dod, q=qd, k=kd, v=vd, out=od, logsumexp=lsed,
    causal=False, softmax_scale=scale, block_q=64, block_k=64,
    use_cuda_kernel=False,
)
t3 = time.perf_counter()

cuda_ms = (t1 - t0) * 1000.0
ref_ms = (t3 - t2) * 1000.0
speedup = ref_ms / cuda_ms if cuda_ms > 0 else float("inf")
print(f"[PERF] shape=(B={B},H={H},T={T},D={D})")
print(f"[PERF] cuda_ms={cuda_ms:.3f}")
print(f"[PERF] ref_ms={ref_ms:.3f}")
print(f"[PERF] speedup_ref_over_cuda={speedup:.2f}x")
PY

echo "[INFO] End: $(date)"
