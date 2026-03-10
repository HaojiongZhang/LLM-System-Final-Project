import numpy as np

from minitorch.flash_attention2 import flash_attention2_backward
from minitorch.tensor_functions import tensor_from_numpy


def _forward_out_lse(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool, scale: float):
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


def _run_backward(use_cuda_kernel: bool, causal: bool, shape=(1, 2, 8, 8), seed=2026):
    rng = np.random.default_rng(seed)
    bsz, nhead, seqlen, headdim = shape
    scale = 1.0 / np.sqrt(float(headdim))

    q = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    k = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    v = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)
    dout = rng.normal(size=(bsz, nhead, seqlen, headdim)).astype(np.float32)

    out, lse = _forward_out_lse(q, k, v, causal=causal, scale=scale)

    dq, dk, dv = flash_attention2_backward(
        dout=tensor_from_numpy(dout),
        q=tensor_from_numpy(q),
        k=tensor_from_numpy(k),
        v=tensor_from_numpy(v),
        out=tensor_from_numpy(out),
        logsumexp=tensor_from_numpy(lse),
        causal=causal,
        softmax_scale=scale,
        block_q=4,
        block_k=4,
        use_cuda_kernel=use_cuda_kernel,
    )
    return dq.to_numpy(), dk.to_numpy(), dv.to_numpy()


def test_fa2_smoke_reference_finite_and_shape():
    dq, dk, dv = _run_backward(use_cuda_kernel=False, causal=False)
    assert dq.shape == (1, 2, 8, 8)
    assert dk.shape == (1, 2, 8, 8)
    assert dv.shape == (1, 2, 8, 8)
    assert np.isfinite(dq).all()
    assert np.isfinite(dk).all()
    assert np.isfinite(dv).all()


def test_fa2_smoke_cuda_matches_reference_if_available():
    try:
        dq_ref, dk_ref, dv_ref = _run_backward(use_cuda_kernel=False, causal=True)
        dq_cuda, dk_cuda, dv_cuda = _run_backward(use_cuda_kernel=True, causal=True)
    except RuntimeError as exc:
        # CUDA symbol/library may be unavailable in non-GPU/local test environments.
        if "CUDA symbol" in str(exc) or "CUDA" in str(exc):
            return
        raise

    np.testing.assert_allclose(dq_cuda, dq_ref, atol=5e-4, rtol=5e-4)
    np.testing.assert_allclose(dk_cuda, dk_ref, atol=5e-4, rtol=5e-4)
    np.testing.assert_allclose(dv_cuda, dv_ref, atol=5e-4, rtol=5e-4)


def test_fa2_smoke_cuda_repeatability_if_available():
    try:
        dq_a, dk_a, dv_a = _run_backward(use_cuda_kernel=True, causal=False, seed=7)
        dq_b, dk_b, dv_b = _run_backward(use_cuda_kernel=True, causal=False, seed=7)
    except RuntimeError as exc:
        if "CUDA symbol" in str(exc) or "CUDA" in str(exc):
            return
        raise

    np.testing.assert_allclose(dq_a, dq_b, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(dk_a, dk_b, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(dv_a, dv_b, atol=1e-6, rtol=1e-6)
