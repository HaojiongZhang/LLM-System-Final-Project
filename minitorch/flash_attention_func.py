"""
FlashAttentionFunc wires FA2 backward into the
autograd graph.

Forward  : computes standard scaled dot-product attention using NumPy and
           saves the logsumexp needed for FA2 backward recomputation.
Backward : calls flash_attention2_backward() (blocked, no O(T^2) storage)
           instead of propagating through the materialised attention matrix.

Shape contract (all tensors passed to apply):
    q, k, v : (B, H, T, D)   k is NOT transposed (un-transpose kT first)
    out      : (B, H, T, D)  returned by forward
    lse      : (B, H, T)     logsumexp per row, saved in ctx

Integration note:
    MultiHeadAttention.self_attention() un-transposes kT to k before calling
    FlashAttentionFunc.make(...).apply(q, k, v).
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np

from .autodiff import Context
from .tensor_functions import Function, tensor_from_numpy

if TYPE_CHECKING:
    from .tensor import Tensor


class FlashAttentionFunc(Function):
    """
    minitorch Function whose forward computes attention + logsumexp and whose
    backward uses the FA2 blocked algorithm (no O(T²) intermediate storage).

    Usage::

        BoundFn = FlashAttentionFunc.make(causal=True, scale=1/sqrt(D))
        out_bhsd = BoundFn.apply(q, k, v)   # (B, H, T, D)
    """

    # Subclass-level config set by make(); read inside _forward via cls.
    _causal: bool  = False
    _scale:  float = 1.0

    # ------------------------------------------------------------------
    # Factory binds causal and scale without extra Tensor args.
    # ------------------------------------------------------------------

    @classmethod
    def make(cls, causal: bool, scale: float) -> type:
        """Return a one-off subclass with causal and scale baked in."""
        return type(
            "_BoundFlashAttn",
            (cls,),
            {"_causal": causal, "_scale": scale},
        )

    # ------------------------------------------------------------------
    # Override _forward to copy class-level config into ctx before forward
    # ------------------------------------------------------------------

    @classmethod
    def _forward(cls, ctx: Context, *inps: "Tensor") -> "Tensor":
        ctx._causal = cls._causal
        ctx._scale  = cls._scale
        return cls.forward(ctx, *inps)

    # ------------------------------------------------------------------
    # Forward: scores to logsumexp to P to out (NumPy, float32).
    # ------------------------------------------------------------------

    @staticmethod
    def forward(ctx: Context, q: "Tensor", k: "Tensor", v: "Tensor") -> "Tensor":
        """
        Args:
            q, k, v : minitorch Tensors of shape (B, H, T, D)
        Returns:
            out     : minitorch Tensor of shape (B, H, T, D)
        """
        scale  = ctx._scale
        causal = ctx._causal

        q_np = np.ascontiguousarray(q.to_numpy(), dtype=np.float32)
        k_np = np.ascontiguousarray(k.to_numpy(), dtype=np.float32)
        v_np = np.ascontiguousarray(v.to_numpy(), dtype=np.float32)

        # scores: (B, H, T, T)
        scores = scale * (q_np @ k_np.transpose(0, 1, 3, 2))

        if causal:
            T = scores.shape[-1]
            mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * (-1e9)
            scores = scores + mask  # broadcast over B, H

        # logsumexp per query row: (B, H, T)
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_s      = np.exp(scores - scores_max)
        sum_exp    = exp_s.sum(axis=-1, keepdims=True)
        lse        = scores_max + np.log(sum_exp)          # (B, H, T, 1)
        lse        = lse[..., 0]                           # (B, H, T)

        # attention weights and output
        P   = exp_s / sum_exp                              # (B, H, T, T)
        out = (P @ v_np).astype(np.float32)               # (B, H, T, D)

        backend = q.backend
        out_t = tensor_from_numpy(np.ascontiguousarray(out), backend=backend)
        lse_t = tensor_from_numpy(np.ascontiguousarray(lse), backend=backend)

        # Save inputs and forward outputs for backward.
        ctx.save_for_backward(q, k, v, out_t, lse_t)

        return out_t

    # ------------------------------------------------------------------
    # Backward: FA2 blocked algorithm with no O(T^2) storage.
    # ------------------------------------------------------------------

    @staticmethod
    def backward(
        ctx: Context, dout: "Tensor"
    ) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """
        Returns (dq, dk, dv), each of shape (B, H, T, D).
        """
        from .flash_attention2 import flash_attention2_backward

        q, k, v, out, lse = ctx.saved_values

        dq, dk, dv = flash_attention2_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            logsumexp=lse,
            causal=ctx._causal,
            softmax_scale=ctx._scale,
        )
        return dq, dk, dv
