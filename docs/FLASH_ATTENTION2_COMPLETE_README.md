# FlashAttention-2 (Python + CUDA) Complete README

This is the complete implementation guide for the FlashAttention-2 backward path in this repository.

Scope:
- Python API and blocked reference implementation
- CUDA wrapper and CUDA kernel entrypoint
- Build and runtime workflow (local + PSC SLURM)
- File-by-file architecture and usage

---

## 1) Quick start

### A. CPU/reference run
Use blocked NumPy reference path for correctness:
- call `flash_attention2_backward(..., use_cuda_kernel=False)`

### B. CUDA run
1. Build kernel shared library:
   - `module load cuda`
   - `bash ./compile_cuda.sh`
2. Call:
   - `flash_attention2_backward(..., use_cuda_kernel=True)`

### C. Batch run on PSC
Use:
- `run_fa2_cuda_tests.slurm`

Submit:
- `sbatch run_fa2_cuda_tests.slurm`

---

## 2) Tensor contracts

### Required tensor shapes
- `q, k, v, out, dout`: `(B, H, T, D)`
- `logsumexp`: `(B, H, T)` or `(B, H, T, 1)`

Where:
- `B`: batch size
- `H`: number of attention heads
- `T`: sequence length
- `D`: head dimension

### Scaling
- `softmax_scale` should match forward exactly.
- default is `1 / sqrt(D)` when unspecified.

---

## 3) Math used in backward

For maximum compatibility across Markdown renderers (GitHub, IDE preview, docs sites),
the equations are written in plain text:

```text
S = scale * (Q K^T) + mask
P = softmax(S)
O = P V

Given dO:
dV = P^T dO
dP = dO V^T
D_i = dot(dO_i, O_i)
dS = P * (dP - D)
dQ = scale * dS K
dK = scale * dS^T Q
```

---

## 4) File-by-file architecture

### 4.1 `minitorch/flash_attention2.py`
Primary API and reference implementation.

Key elements:
- `FlashAttention2ForwardContext`
  - lightweight structure for forward/backward integration
- `_validate_attention_shapes(...)`
  - enforces shape/rank contract
- `flash_attention2_backward(...)`
  - dispatches to CUDA path or blocked NumPy path
- `flash_attention2_backward_from_context(...)`
  - convenience API for future forward integration

How it works:
1. optional CUDA dispatch
2. convert tensors to contiguous NumPy
3. blocked loops over query/key tiles
4. recompute local probabilities from `logsumexp`
5. accumulate `dq/dk/dv`
6. convert back to miniTorch tensors

---

### 4.2 `minitorch/cuda_kernel_ops.py`
Python-to-CUDA bridge through `ctypes`.

Key elements:
- `HAS_FLASH_ATTENTION2_BWD`
  - checks whether `combine.so` exports `FlashAttention2Backward`
- `CudaKernelOps.flash_attention2_backward(...)`
  - validates shapes
  - canonicalizes `logsumexp`
  - sets C ABI argtypes
  - calls C wrapper
  - wraps outputs to tensors

How it works:
- Python passes contiguous 1D views + explicit dimensions `(B,H,T,D)`.
- C wrapper handles H2D copy, kernel launch, D2H copy.

---

### 4.3 `src/combine.cu`
CUDA kernel implementation + C ABI wrapper.

Key elements:
- `FlashAttention2BackwardKernel(...)`
  - one block per query row `(b,h,i)`
  - key/value dimension processed in shared-memory tiles (`BK=16`)
  - cooperative reductions inside block for:
    - score dot product `dot(q_i, k_j)`
    - `dP_ij = dot(dO_i, V_j)`
    - row scalar `D_i = dot(dO_i, O_i)`
  - `dQ` accumulated without atomics (row-private block ownership)
  - `dK` / `dV` use atomic add only where cross-row write conflicts exist
- `extern "C" void FlashAttention2Backward(...)`
  - entrypoint called by `ctypes`
  - alloc/copy/launch/copy-back/free

Notes:
- this implementation is now a practical tiled kernel (not a naive pairwise kernel)
- shared memory is used to leverage locality and reduce global-memory traffic
- selective atomics are kept only for unavoidable accumulation conflicts

---

### 4.4 `compile_cuda.sh`
Build script for CUDA shared library:
- outputs `minitorch/cuda_kernels/combine.so`

---

### 4.5 `tests/test_flash_attention2_backward.py`
Correctness and API contract tests.

Covers:
- non-causal and causal parity with dense reference
- non-divisible tile sizes
- accepted `(B,H,T,1)` `logsumexp`
- invalid shape/tile rejection
- forward-context helper equivalence

---

### 4.6 `run_fa2_cuda_tests.slurm`
PSC GPU batch validation script.

Pipeline:
1. load CUDA module
2. activate venv
3. build shared library
4. verify symbol
5. run FA2 tests
6. run explicit CUDA runtime check

---

## 5) Usage examples

### A. Direct API
- prepare `q,k,v,out,logsumexp,dout`
- call `flash_attention2_backward(...)`

### B. Context API
- forward stores `FlashAttention2ForwardContext`
- backward calls `flash_attention2_backward_from_context(...)`

### C. Force path
- CUDA: `use_cuda_kernel=True`
- reference: `use_cuda_kernel=False`

---

## 6) Typical integration sequence

1. forward computes and stores:
   - `out`
   - `logsumexp`
   - `causal`
   - `softmax_scale`
2. backward receives `dout`
3. call FA2 backward (direct or context helper)
4. consume `dq, dk, dv` in autograd graph

---

## 7) Troubleshooting

### `FlashAttention2Backward CUDA symbol not found`
- rebuild with `bash ./compile_cuda.sh`
- verify export with `nm -D minitorch/cuda_kernels/combine.so | grep FlashAttention2Backward`

### GPU not visible in shell
- use GPU allocation (`salloc`/`sbatch`)
- load CUDA module before building/running

### Numerical mismatch
- ensure forward/backward use identical:
  - `causal`
  - `softmax_scale`
  - `logsumexp` from the same forward run

---

## 8) Latest GPU validation (PSC)

Latest validation job:
- Job ID: `37923424`
- State: `COMPLETED`
- ExitCode: `0:0`
- Log: `fa2_cuda_test_37923424.log`

Observed outputs:
- FA2 backward tests: `6 passed`
- Explicit CUDA path check: success
- Timing probe (single-run, includes Python + wrapper overhead):
  - shape `(B=2,H=4,T=128,D=64)`
  - `cuda_ms=2525.329`
  - `ref_ms=2489.146`
  - speedup ratio (`ref/cuda`) `= 0.99x`

Interpretation:
- kernel is now tiled and hardware-aware,
- but end-to-end runtime is still dominated by host-device copy overhead and
  remaining atomic accumulation on `dK/dV` in this wrapper path.

---

## 9) Related docs

- `docs/FLASH_ATTENTION2_BACKWARD.md`
- `docs/FLASH_ATTENTION2_BACKWARD_zh-TW.md`
- `docs/FLASH_ATTENTION2_BACKWARD_DEVELOPER_GUIDE.md`
- `docs/FLASH_ATTENTION2_BACKWARD_DEVELOPER_GUIDE_zh-TW.md`
- `docs/FLASH_ATTENTION2_BACKWARD_CUDA.md`
- `docs/FLASH_ATTENTION2_BACKWARD_CUDA_zh-TW.md`
