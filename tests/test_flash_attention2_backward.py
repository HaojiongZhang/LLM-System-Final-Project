import numpy as np

from minitorch.flash_attention2 import (
    FlashAttention2ForwardContext,
    flash_attention2_backward,
    flash_attention2_backward_from_context,
)
from minitorch.tensor_functions import tensor_from_numpy


# Test tensor convention used throughout this file:
# q,k,v,out,dout,dq,dk,dv: (B,H,T,D)
# lse:                     (B,H,T) or (B,H,T,1)


def _reference_backward(
    dout: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    out: np.ndarray,
    causal: bool,
    scale: float,
):
    """Reference dense attention backward for correctness checks.

    This function intentionally materializes full attention probabilities `P`
    and serves as a numerical ground truth for the blocked implementation.
    """
    bsz, nhead, seqlen, headdim = q.shape

    dq = np.zeros_like(q, dtype=np.float32)
    dk = np.zeros_like(k, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)

    for b in range(bsz):
        for h in range(nhead):
            scores = (q[b, h] @ k[b, h].T) * scale

            if causal:
                row = np.arange(seqlen)[:, None]
                col = np.arange(seqlen)[None, :]
                scores = np.where(col > row, -np.inf, scores)

            # Stable softmax.
            m = np.max(scores, axis=-1, keepdims=True)
            p = np.exp(scores - m)
            p = p / np.sum(p, axis=-1, keepdims=True)

            do = dout[b, h]
            o = out[b, h]

            dV = p.T @ do
            dP = do @ v[b, h].T
            D = np.sum(do * o, axis=-1, keepdims=True)
            dS = p * (dP - D)
            dQ = (dS @ k[b, h]) * scale
            dK = (dS.T @ q[b, h]) * scale

            dq[b, h] = dQ
            dk[b, h] = dK
            dv[b, h] = dV

    return dq, dk, dv


def _forward_out_and_lse(q, k, v, causal, scale):
    """Compute `out` and `logsumexp` needed by backward API.

    We compute stable softmax via max-shift and keep per-row log-sum-exp.
    """
    bsz, nhead, seqlen, _ = q.shape
    out = np.zeros_like(q, dtype=np.float32)
    lse = np.zeros((bsz, nhead, seqlen), dtype=np.float32)

    for b in range(bsz):
        for h in range(nhead):
            scores = (q[b, h] @ k[b, h].T) * scale
            if causal:
                row = np.arange(seqlen)[:, None]
                col = np.arange(seqlen)[None, :]
                scores = np.where(col > row, -np.inf, scores)

            m = np.max(scores, axis=-1, keepdims=True)
            ex = np.exp(scores - m)
            z = np.sum(ex, axis=-1, keepdims=True)
            p = ex / z

            out[b, h] = p @ v[b, h]
            lse[b, h] = (np.log(z) + m)[:, 0]

    return out, lse


def _run_case(causal: bool):
    # Moderate shape that exercises multi-batch/multi-head behavior.
    rng = np.random.default_rng(2026)
    bsz, nhead, seqlen, headdim = 2, 3, 16, 8
    scale = 1.0 / np.sqrt(float(headdim))

    q = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    k = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    v = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    dout = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)

    out, lse = _forward_out_and_lse(q, k, v, causal=causal, scale=scale)

    dq_ref, dk_ref, dv_ref = _reference_backward(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        causal=causal,
        scale=scale,
    )

    # Use non-divisible block sizes to verify tile boundary correctness.
    dq, dk, dv = flash_attention2_backward(
        dout=tensor_from_numpy(dout),
        q=tensor_from_numpy(q),
        k=tensor_from_numpy(k),
        v=tensor_from_numpy(v),
        out=tensor_from_numpy(out),
        logsumexp=tensor_from_numpy(lse),
        causal=causal,
        softmax_scale=scale,
        block_q=7,
        block_k=5,
    )

    np.testing.assert_allclose(dq.to_numpy(), dq_ref, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(dk.to_numpy(), dk_ref, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(dv.to_numpy(), dv_ref, atol=3e-5, rtol=3e-5)


def test_flash_attention2_backward_noncausal():
    _run_case(causal=False)


def test_flash_attention2_backward_causal():
    _run_case(causal=True)


def test_flash_attention2_backward_accepts_lse_rank4():
    """`logsumexp` with trailing singleton dim should be accepted."""
    rng = np.random.default_rng(7)
    bsz, nhead, seqlen, headdim = 1, 2, 8, 4
    scale = 1.0 / np.sqrt(float(headdim))

    q = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    k = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    v = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    dout = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)

    out, lse = _forward_out_and_lse(q, k, v, causal=False, scale=scale)
    lse_rank4 = lse[..., None]

    dq, dk, dv = flash_attention2_backward(
        dout=tensor_from_numpy(dout),
        q=tensor_from_numpy(q),
        k=tensor_from_numpy(k),
        v=tensor_from_numpy(v),
        out=tensor_from_numpy(out),
        logsumexp=tensor_from_numpy(lse_rank4),
        causal=False,
        softmax_scale=scale,
        block_q=4,
        block_k=4,
    )

    # Sanity checks: outputs exist and shape-preserving.
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape


def test_flash_attention2_backward_rejects_nonpositive_block_size():
    """Tile sizes must be positive by API contract."""
    rng = np.random.default_rng(11)
    bsz, nhead, seqlen, headdim = 1, 1, 4, 4

    q = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    k = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    v = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    dout = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    out, lse = _forward_out_and_lse(q, k, v, causal=False, scale=0.5)

    try:
        flash_attention2_backward(
            dout=tensor_from_numpy(dout),
            q=tensor_from_numpy(q),
            k=tensor_from_numpy(k),
            v=tensor_from_numpy(v),
            out=tensor_from_numpy(out),
            logsumexp=tensor_from_numpy(lse),
            block_q=0,
            block_k=4,
        )
        assert False, "Expected ValueError for block_q=0"
    except ValueError as exc:
        assert "must be positive" in str(exc)


def test_flash_attention2_backward_rejects_bad_lse_shape():
    """Shape validation should fail fast for malformed `logsumexp`."""
    rng = np.random.default_rng(13)
    bsz, nhead, seqlen, headdim = 1, 1, 6, 4

    q = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    k = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    v = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    dout = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    out, _ = _forward_out_and_lse(q, k, v, causal=False, scale=0.5)

    bad_lse = rng.normal(size=(bsz, nhead, seqlen, 2)).astype(np.float32)

    try:
        flash_attention2_backward(
            dout=tensor_from_numpy(dout),
            q=tensor_from_numpy(q),
            k=tensor_from_numpy(k),
            v=tensor_from_numpy(v),
            out=tensor_from_numpy(out),
            logsumexp=tensor_from_numpy(bad_lse),
            block_q=4,
            block_k=4,
        )
        assert False, "Expected ValueError for malformed logsumexp shape"
    except ValueError as exc:
        assert "logsumexp" in str(exc)


def test_flash_attention2_backward_from_context_matches_direct_call():
    """Forward-context helper should be numerically identical to direct API."""
    rng = np.random.default_rng(17)
    bsz, nhead, seqlen, headdim = 1, 2, 10, 8
    scale = 1.0 / np.sqrt(float(headdim))

    q = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    k = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    v = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    dout = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    out, lse = _forward_out_and_lse(q, k, v, causal=True, scale=scale)

    dq_a, dk_a, dv_a = flash_attention2_backward(
        dout=tensor_from_numpy(dout),
        q=tensor_from_numpy(q),
        k=tensor_from_numpy(k),
        v=tensor_from_numpy(v),
        out=tensor_from_numpy(out),
        logsumexp=tensor_from_numpy(lse),
        causal=True,
        softmax_scale=scale,
        block_q=4,
        block_k=3,
        use_cuda_kernel=False,
    )

    ctx = FlashAttention2ForwardContext(
        out=tensor_from_numpy(out),
        logsumexp=tensor_from_numpy(lse),
        causal=True,
        softmax_scale=scale,
        block_q=4,
        block_k=3,
    )

    dq_b, dk_b, dv_b = flash_attention2_backward_from_context(
        dout=tensor_from_numpy(dout),
        q=tensor_from_numpy(q),
        k=tensor_from_numpy(k),
        v=tensor_from_numpy(v),
        ctx=ctx,
        use_cuda_kernel=False,
    )

    np.testing.assert_allclose(dq_a.to_numpy(), dq_b.to_numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(dk_a.to_numpy(), dk_b.to_numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(dv_a.to_numpy(), dv_b.to_numpy(), atol=1e-6, rtol=1e-6)
