"""
FlashAttention-2 backward (educational implementation).

This module provides a **blocked backward pass** for scaled dot-product attention.
It is designed to be easy to read and verify, not to be the fastest possible kernel.

Why this exists in this repository
----------------------------------
The project already has a full miniTorch stack and a standard attention path.
This file adds a standalone FA2-style backward routine that:
1. avoids materializing the full attention matrix in memory,
2. recomputes local softmax probabilities per tile from `q`, `k`, and `logsumexp`,
3. accumulates gradients for `q`, `k`, and `v` tile-by-tile.

Important scope note
--------------------
- This is **backward only**.
- It assumes a separate forward pass produced `out` and per-row `logsumexp`.
- It runs in NumPy for clarity and correctness, then converts results back to
  miniTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .tensor import Tensor
from .tensor_functions import tensor_from_numpy


@dataclass(frozen=True)
class FlashAttention2ForwardContext:
    """Minimal forward context consumed by FA2 backward.

    This class is intentionally small so future forward implementations can
    construct it without coupling to a specific kernel implementation.
    """

    # `out`: forward output tensor O = softmax(QK^T)V, shape (B,H,T,D).
    out: Tensor
    # `logsumexp`: row-wise log-sum-exp used to reconstruct probabilities,
    # shape (B,H,T) or (B,H,T,1).
    logsumexp: Tensor
    # `causal`: whether forward used causal mask (j > i masked).
    causal: bool = False
    # `softmax_scale`: exact scale used by forward, typically 1/sqrt(D).
    softmax_scale: float | None = None
    # Tile sizes used by blocked backward reference path.
    block_q: int = 64
    block_k: int = 64


def _validate_attention_shapes(
    dout_np: np.ndarray,
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    out_np: np.ndarray,
    lse_np: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Validate tensor shapes and return `(B, H, T, D)`.

    Expected shapes:
    - `q`, `k`, `v`, `out`, `dout`: `(B, H, T, D)`
    - `logsumexp`: `(B, H, T)` or `(B, H, T, 1)`
    """
    # Rank contract: FA2 backward here expects dense 4D tensors (B,H,T,D).
    if q_np.ndim != 4:
        raise ValueError(f"`q` must be rank-4 `(B, H, T, D)`, got {q_np.shape}")

    b, h, t, d = q_np.shape

    # All activations/gradients share identical shape.
    expected = (b, h, t, d)
    for name, arr in (
        ("k", k_np),
        ("v", v_np),
        ("out", out_np),
        ("dout", dout_np),
    ):
        if arr.shape != expected:
            raise ValueError(f"`{name}` must have shape {expected}, got {arr.shape}")

    # Allow an optional trailing singleton dim for convenience with callers
    # that preserve keepdim=True semantics from reductions.
    if lse_np.shape == (b, h, t, 1):
        # Canonicalize to (B, H, T)
        lse_np = lse_np[..., 0]
    if lse_np.shape != (b, h, t):
        raise ValueError(
            "`logsumexp` must have shape (B, H, T) or (B, H, T, 1), "
            f"got {lse_np.shape}"
        )

    return b, h, t, d


def flash_attention2_backward(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    block_q: int = 64,
    block_k: int = 64,
    use_cuda_kernel: bool | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute FA2-style backward pass for scaled dot-product attention.

    Mathematical setup
    ------------------
    Let
    - `S = scale * Q K^T + mask`
    - `P = softmax(S)`
    - `O = P V`

    Given `dO`, we compute:
    - `dV = P^T dO`
    - `dP = dO V^T`
    - `D_i = <dO_i, O_i>` (row-wise dot product)
    - `dS = P * (dP - D)`
    - `dQ = scale * dS K`
    - `dK = scale * dS^T Q`

    FA2-style memory behavior
    -------------------------
    We do *blocked recomputation* of `P` from `Q`, `K`, and `logsumexp`:
    - iterate over query blocks (`block_q`),
    - iterate over key/value blocks (`block_k`),
    - never store a full `(T, T)` attention matrix.

    Parameters
    ----------
    dout, q, k, v, out : Tensor
        All expected to have shape `(B, H, T, D)`.
    logsumexp : Tensor
        Shape `(B, H, T)` or `(B, H, T, 1)`, produced by forward as:
        `logsumexp_i = log(sum_j(exp(score_ij)))` after masking.
    causal : bool
        If True, applies causal masking (`j > i` is masked).
    softmax_scale : float | None
        Scaling factor used in forward. Defaults to `1 / sqrt(D)` if None.
    block_q, block_k : int
        Tile sizes for query and key/value dimensions.
    use_cuda_kernel : bool | None
        Controls CUDA-kernel dispatch.
        - `None` (default): auto-dispatch on CUDA backend when available.
        - `True`: force CUDA kernel path; raises if unavailable.
        - `False`: force NumPy blocked reference path.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        `(dq, dk, dv)` with same shape/backend as inputs.

    Notes
    -----
    - Uses NumPy internally for readability.
    - Returns miniTorch tensors via `tensor_from_numpy`.
    - This function is intentionally explicit and heavily commented for study.
    """
    # ---------------------------------------------------------------------
    # 0) Optional CUDA dispatch.
    # ---------------------------------------------------------------------
    # Auto mode: only dispatch CUDA when current backend is CUDA-capable.
    if use_cuda_kernel is None:
        use_cuda_kernel = bool(getattr(q.backend, "cuda", False))

    # CUDA path delegates math to compiled kernel wrapper in
    # `minitorch/cuda_kernel_ops.py`.
    if use_cuda_kernel:
        from .cuda_kernel_ops import CudaKernelOps  # Local import to avoid cycles.

        if not hasattr(CudaKernelOps, "flash_attention2_backward"):
            raise RuntimeError("CUDA kernel path requested but not available")

        return CudaKernelOps.flash_attention2_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            logsumexp=logsumexp,
            causal=causal,
            softmax_scale=softmax_scale,
        )

    # ---------------------------------------------------------------------
    # 1) Move to contiguous NumPy arrays for explicit blocked math.
    # ---------------------------------------------------------------------
    # NumPy reference path uses contiguous float32 arrays so all matmul/slicing
    # operations are explicit and deterministic.
    dout_np = np.ascontiguousarray(dout.to_numpy(), dtype=np.float32)
    q_np = np.ascontiguousarray(q.to_numpy(), dtype=np.float32)
    k_np = np.ascontiguousarray(k.to_numpy(), dtype=np.float32)
    v_np = np.ascontiguousarray(v.to_numpy(), dtype=np.float32)
    out_np = np.ascontiguousarray(out.to_numpy(), dtype=np.float32)
    lse_np = np.ascontiguousarray(logsumexp.to_numpy(), dtype=np.float32)

    bsz, nhead, seqlen, headdim = _validate_attention_shapes(
        dout_np, q_np, k_np, v_np, out_np, lse_np
    )

    # Canonicalize lse to shape (B,H,T) for simple broadcasting in `exp` step.
    if lse_np.ndim == 4:
        lse_np = lse_np[..., 0]

    if softmax_scale is None:
        softmax_scale = 1.0 / np.sqrt(float(headdim))

    # Defensive checks on tile sizes.
    if block_q <= 0 or block_k <= 0:
        raise ValueError("`block_q` and `block_k` must be positive")

    # ---------------------------------------------------------------------
    # 2) Allocate gradient buffers.
    # ---------------------------------------------------------------------
    # Gradient buffers use same layout as inputs for direct indexed updates.
    dq = np.zeros_like(q_np, dtype=np.float32)
    dk = np.zeros_like(k_np, dtype=np.float32)
    dv = np.zeros_like(v_np, dtype=np.float32)

    # ---------------------------------------------------------------------
    # 3) Main FA2 backward loops.
    # ---------------------------------------------------------------------
    # We iterate per (batch, head), then over q-tiles and k/v-tiles.
    # This mirrors FlashAttention's idea: stream blocks through SRAM/registers
    # rather than materializing full attention matrices.
    # ---------------------------------------------------------------------
    for b in range(bsz):
        for h in range(nhead):
            # Row-wise scalar term D_i = <dO_i, O_i>, shape (T,).
            # Reused across all key blocks for the same query row.
            d_row = np.sum(dout_np[b, h] * out_np[b, h], axis=-1)

            # Query-block loop.
            for q_start in range(0, seqlen, block_q):
                q_end = min(q_start + block_q, seqlen)

                # Tile views for this query block.
                q_tile = q_np[b, h, q_start:q_end, :]          # (BQ, D)
                do_tile = dout_np[b, h, q_start:q_end, :]      # (BQ, D)
                lse_tile = lse_np[b, h, q_start:q_end]         # (BQ,)
                d_tile = d_row[q_start:q_end]                  # (BQ,)

                # Key/value-block loop.
                for k_start in range(0, seqlen, block_k):
                    k_end = min(k_start + block_k, seqlen)

                    k_tile = k_np[b, h, k_start:k_end, :]      # (BK, D)
                    v_tile = v_np[b, h, k_start:k_end, :]      # (BK, D)

                    # -----------------------------------------------------
                    # Recompute local scores and local probabilities.
                    # scores shape: (BQ, BK)
                    # -----------------------------------------------------
                    scores = (q_tile @ k_tile.T) * softmax_scale

                    if causal:
                        # Global row and col indices for this tile.
                        # Causal rule: query position i cannot attend to key j when j > i.
                        q_idx = np.arange(q_start, q_end)[:, None]  # (BQ, 1)
                        k_idx = np.arange(k_start, k_end)[None, :]  # (1, BK)
                        scores = np.where(k_idx > q_idx, -np.inf, scores)

                    # p_ij = exp(score_ij - logsumexp_i).
                    # lse_tile has shape (BQ,), broadcast to (BQ, BK).
                    p = np.exp(scores - lse_tile[:, None]).astype(np.float32)

                    # -----------------------------------------------------
                    # dV update: dV += P^T @ dO
                    # -----------------------------------------------------
                    dv[b, h, k_start:k_end, :] += p.T @ do_tile

                    # -----------------------------------------------------
                    # dS construction via softmax Jacobian trick:
                    # dP = dO @ V^T
                    # dS = P * (dP - D)
                    # -----------------------------------------------------
                    dP = do_tile @ v_tile.T
                    dS = p * (dP - d_tile[:, None])

                    # -----------------------------------------------------
                    # dQ and dK updates.
                    # -----------------------------------------------------
                    dq[b, h, q_start:q_end, :] += (dS @ k_tile) * softmax_scale
                    dk[b, h, k_start:k_end, :] += (dS.T @ q_tile) * softmax_scale

    # ---------------------------------------------------------------------
    # 4) Convert back to miniTorch tensors on original backend.
    # ---------------------------------------------------------------------
    # Preserve original backend contract for callers.
    backend = q.backend
    dq_t = tensor_from_numpy(np.ascontiguousarray(dq), backend=backend, requires_grad=False)
    dk_t = tensor_from_numpy(np.ascontiguousarray(dk), backend=backend, requires_grad=False)
    dv_t = tensor_from_numpy(np.ascontiguousarray(dv), backend=backend, requires_grad=False)

    return dq_t, dk_t, dv_t


def flash_attention2_backward_from_context(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ctx: FlashAttention2ForwardContext,
    use_cuda_kernel: bool | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward helper for future forward integration.

    Forward implementations can cache `FlashAttention2ForwardContext` and pass
    it directly here during backward.
    """
    return flash_attention2_backward(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=ctx.out,
        logsumexp=ctx.logsumexp,
        causal=ctx.causal,
        softmax_scale=ctx.softmax_scale,
        block_q=ctx.block_q,
        block_k=ctx.block_k,
        use_cuda_kernel=use_cuda_kernel,
    )
