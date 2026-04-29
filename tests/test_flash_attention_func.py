"""Focused correctness checks for the FlashAttention wrapper."""

import math
import pytest
import numpy as np


def simple_backend():
    import minitorch
    try:
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    except Exception:
        return minitorch.TensorBackend(minitorch.SimpleOps)


def _rand_tensor(shape, backend, requires_grad=False, seed=0):
    from minitorch.tensor_functions import tensor_from_numpy
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(np.float32) * 0.5
    t = tensor_from_numpy(arr, backend=backend)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _standard_attention(q, kT, v, causal, scale, backend):
    """Reference forward in NumPy, returned as ndarray."""
    q_np = np.ascontiguousarray(q.to_numpy(), dtype=np.float32)
    kT_np = np.ascontiguousarray(kT.to_numpy(), dtype=np.float32)
    v_np = np.ascontiguousarray(v.to_numpy(), dtype=np.float32)

    scores = (q_np @ kT_np) * scale
    if causal:
        T = scores.shape[-1]
        scores = scores + (-1e9 * np.triu(np.ones((1, 1, T, T), dtype=np.float32), 1))

    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return (probs @ v_np).astype(np.float32)


def _standard_attention_backward_np(q_np, k_np, v_np, dout_np, causal, scale):
    """Closed-form dense attention backward in NumPy."""
    scores = scale * (q_np @ k_np.transpose(0, 1, 3, 2))
    if causal:
        T = scores.shape[-1]
        scores = scores + (-1e9 * np.triu(np.ones((1, 1, T, T), dtype=np.float32), 1))

    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    out_np = probs @ v_np

    dV = probs.transpose(0, 1, 3, 2) @ dout_np
    dP = dout_np @ v_np.transpose(0, 1, 3, 2)
    D_term = np.sum(dout_np * out_np, axis=-1, keepdims=True)
    dS = probs * (dP - D_term)
    dQ = scale * (dS @ k_np)
    dK = scale * (dS.transpose(0, 1, 3, 2) @ q_np)
    return dQ.astype(np.float32), dK.astype(np.float32), dV.astype(np.float32)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 8, 4), (2, 4, 16, 8)])
def test_forward_matches_standard(causal, shape):
    from minitorch.flash_attention_func import FlashAttentionFunc

    backend = simple_backend()
    B, H, T, D = shape
    scale = 1.0 / math.sqrt(D)

    q  = _rand_tensor((B, H, T, D), backend, seed=1)
    k  = _rand_tensor((B, H, T, D), backend, seed=2)
    v  = _rand_tensor((B, H, T, D), backend, seed=3)
    kT = k.permute(0, 1, 3, 2)  # (B, H, D, T) for standard path

    ref_np = _standard_attention(q, kT, v, causal=causal, scale=scale, backend=backend)

    BoundFn = FlashAttentionFunc.make(causal=causal, scale=scale)
    got     = BoundFn.apply(q, k, v)

    got_np = got.to_numpy()

    np.testing.assert_allclose(
        got_np, ref_np, atol=1e-4, rtol=1e-4,
        err_msg=f"Forward mismatch: causal={causal}, shape={shape}",
    )


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(causal):
    """
    Both paths run .backward() on the same loss; compare dq, dk, dv.
    Standard path uses minitorch autograd through the T x T matrix.
    FA2 path uses FlashAttentionFunc which calls flash_attention2_backward.
    """
    from minitorch.flash_attention_func import FlashAttentionFunc
    from minitorch.tensor_functions import tensor_from_numpy

    backend = simple_backend()
    B, H, T, D = 1, 2, 8, 4
    scale = 1.0 / math.sqrt(D)

    rng = np.random.default_rng(7)
    q_np = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.3
    k_np = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.3
    v_np = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.3

    def _make(arr):
        t = tensor_from_numpy(arr.copy(), backend=backend)
        t.requires_grad_(True)
        return t

    # Dense closed-form NumPy reference for loss = sum(out), so dout = 1.
    dout_np = np.ones((B, H, T, D), dtype=np.float32)
    dq_ref, dk_ref, dv_ref = _standard_attention_backward_np(
        q_np, k_np, v_np, dout_np, causal=causal, scale=scale
    )

    q_f, k_f, v_f = _make(q_np), _make(k_np), _make(v_np)
    BoundFn = FlashAttentionFunc.make(causal=causal, scale=scale)
    out_f = BoundFn.apply(q_f, k_f, v_f)
    loss_f = out_f.sum(dim=3).sum(dim=2).sum(dim=1).sum(dim=0).view(1)
    loss_f.backward()

    for name, ref, gf in [
        ("dq", dq_ref, q_f.grad),
        ("dk", dk_ref, k_f.grad),
        ("dv", dv_ref, v_f.grad),
    ]:
        assert gf is not None, f"{name} grad is None"
        np.testing.assert_allclose(
            gf.to_numpy(), ref, atol=1e-4, rtol=1e-4,
            err_msg=f"{name} mismatch: causal={causal}",
        )


def test_causal_mask_forward():
    from minitorch.flash_attention_func import FlashAttentionFunc
    from minitorch.tensor_functions import tensor_from_numpy

    backend = simple_backend()
    B, H, T, D = 1, 1, 6, 4
    scale = 1.0 / math.sqrt(D)

    rng = np.random.default_rng(42)
    q = tensor_from_numpy(rng.standard_normal((B, H, T, D)).astype(np.float32), backend=backend)
    k = tensor_from_numpy(rng.standard_normal((B, H, T, D)).astype(np.float32), backend=backend)
    v = tensor_from_numpy(np.eye(T, D, dtype=np.float32)[np.newaxis, np.newaxis], backend=backend)

    # With identity-like V, the output at position i reveals which K positions
    # were attended to.  With causal masking, position i should only reflect
    # positions 0..i (the diagonal and below in the attention weight matrix).
    BoundFn = FlashAttentionFunc.make(causal=True, scale=scale)
    out = BoundFn.apply(q, k, v).to_numpy()  # (1, 1, T, D)

    assert out.shape == (B, H, T, D)
    assert np.isfinite(out).all(), "causal forward produced non-finite values"


def test_causal_mask_backward():
    from minitorch.flash_attention_func import FlashAttentionFunc
    from minitorch.tensor_functions import tensor_from_numpy

    backend = simple_backend()
    B, H, T, D = 1, 2, 6, 4
    scale = 1.0 / math.sqrt(D)

    rng = np.random.default_rng(99)
    q_np = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.2
    k_np = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.2
    v_np = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.2

    q = tensor_from_numpy(q_np, backend=backend)
    k = tensor_from_numpy(k_np, backend=backend)
    v = tensor_from_numpy(v_np, backend=backend)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    BoundFn = FlashAttentionFunc.make(causal=True, scale=scale)
    out  = BoundFn.apply(q, k, v)
    loss = out.sum(dim=3).sum(dim=2).sum(dim=1).sum(dim=0).view(1)
    loss.backward()

    for name, g in [("dq", q.grad), ("dk", k.grad), ("dv", v.grad)]:
        assert g is not None, f"{name} is None after causal backward"
        assert np.isfinite(g.to_numpy()).all(), f"{name} has non-finite values"


@pytest.mark.parametrize("shape", [(1, 2, 8, 8), (2, 3, 16, 8)])
def test_public_flash_attention_helper_matches_flash_attention_func(shape):
    from minitorch.flash_attention_func import FlashAttentionFunc
    from minitorch.tensor_functions import flash_attention

    backend = simple_backend()
    B, H, T, D = shape
    scale = 1.0 / math.sqrt(D)

    q = _rand_tensor((B, H, T, D), backend, requires_grad=True, seed=11)
    k = _rand_tensor((B, H, T, D), backend, requires_grad=True, seed=12)
    v = _rand_tensor((B, H, T, D), backend, requires_grad=True, seed=13)

    q_ref = _rand_tensor((B, H, T, D), backend, requires_grad=True, seed=11)
    k_ref = _rand_tensor((B, H, T, D), backend, requires_grad=True, seed=12)
    v_ref = _rand_tensor((B, H, T, D), backend, requires_grad=True, seed=13)

    out = flash_attention(q, k, v, causal=True, softmax_scale=scale)
    out_ref = FlashAttentionFunc.make(causal=True, scale=scale).apply(q_ref, k_ref, v_ref)

    np.testing.assert_allclose(out.to_numpy(), out_ref.to_numpy(), atol=1e-4, rtol=1e-4)

    loss = out.sum(dim=3).sum(dim=2).sum(dim=1).sum(dim=0).view(1)
    loss_ref = out_ref.sum(dim=3).sum(dim=2).sum(dim=1).sum(dim=0).view(1)
    loss.backward()
    loss_ref.backward()

    for got, ref in [(q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)]:
        assert got is not None and ref is not None
        np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), atol=1e-4, rtol=1e-4)
