"""End-to-end checks for training, FlashAttention, and paged decoding."""

import pytest
import numpy as np


def simple_backend():
    import minitorch
    try:
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    except Exception:
        return minitorch.TensorBackend(minitorch.SimpleOps)


def _small_model(backend, use_flash_attn=False):
    from minitorch.transformer import DecoderLM
    return DecoderLM(
        n_vocab=32, n_embd=16, n_head=2,
        n_positions=32, p_dropout=0.0,
        backend=backend, use_flash_attn=use_flash_attn,
    )


def _idx(batch=1, seq=8, vocab=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=(batch, seq), dtype=np.int32)


def test_training_step_standard():
    from minitorch.tensor_functions import tensor_from_numpy

    backend = simple_backend()
    model   = _small_model(backend, use_flash_attn=False)
    idx     = tensor_from_numpy(_idx(), backend=backend)

    logits  = model(idx)
    assert logits.shape == (1, 8, 32)

    loss = logits.sum(dim=2).sum(dim=1).sum(dim=0).view(1)
    loss.backward()

    # Check that at least one parameter has a gradient.
    grads = [p.value.grad for p in model.parameters() if p.value.grad is not None]
    assert len(grads) > 0, "No gradients computed in standard training step"


def test_training_step_flash_matches_standard():
    from minitorch.tensor_functions import tensor_from_numpy

    backend = simple_backend()
    idx_np  = _idx(batch=1, seq=8, seed=5)

    losses = {}
    for use_fa2 in (False, True):
        model  = _small_model(backend, use_flash_attn=use_fa2)
        if use_fa2:
            std_params = list(std_model.parameters())
            fa2_params = list(model.parameters())
            for sp, fp in zip(std_params, fa2_params):
                fp.update(sp.value.detach())

        idx    = tensor_from_numpy(idx_np, backend=backend)
        logits = model(idx)
        loss   = logits.sum(dim=2).sum(dim=1).sum(dim=0).view(1)
        losses["FA2" if use_fa2 else "Standard"] = loss.to_numpy().item()

        if not use_fa2:
            std_model = model

    np.testing.assert_allclose(
        losses["FA2"], losses["Standard"], atol=1e-4,
        err_msg="FA2 and standard training losses do not match",
    )


def test_inference_paged_stable():
    from minitorch.tensor_functions import tensor_from_numpy
    from minitorch.paged_attention import BlockManager

    backend = simple_backend()
    model   = _small_model(backend)

    bm = BlockManager(
        num_layers=4, num_blocks=32, block_size=4,
        n_head=2, head_dim=8, backend=backend,
    )

    prompt_np = np.array([[1, 2, 3, 4]], dtype=np.int32)
    prompt    = tensor_from_numpy(prompt_np, backend=backend)

    logits = model.prefill(prompt, seq_id=0, block_manager=bm)
    assert np.isfinite(logits.to_numpy()).all(), "prefill produced non-finite logits"

    last_token = np.array([[int(logits.to_numpy()[0, -1, :].argmax())]], dtype=np.int32)
    for _ in range(5):
        tok   = tensor_from_numpy(last_token, backend=backend)
        logit = model.decode_step(tok, seq_id=0, block_manager=bm)
        assert logit.shape == (1, 1, 32)
        assert np.isfinite(logit.to_numpy()).all(), "decode_step produced non-finite logits"
        last_token = np.array([[int(logit.to_numpy()[0, 0, :].argmax())]], dtype=np.int32)

    bm.free_seq(0)


def test_paged_matches_non_paged():
    from minitorch.tensor_functions import tensor_from_numpy
    from minitorch.paged_attention import BlockManager

    backend = simple_backend()
    model   = _small_model(backend)

    prompt_np = np.array([[5, 6, 7, 8]], dtype=np.int32)

    non_paged_tokens = []
    ctx_np = prompt_np.copy()
    for _ in range(4):
        ctx = tensor_from_numpy(ctx_np, backend=backend)
        logits = model(ctx)         # (1, seq, vocab)
        next_tok = int(logits.to_numpy()[0, -1, :].argmax())
        non_paged_tokens.append(next_tok)
        ctx_np = np.concatenate(
            [ctx_np, np.array([[next_tok]], dtype=np.int32)],
            axis=1,
        )

    paged_tokens = []
    bm = BlockManager(
        num_layers=4, num_blocks=64, block_size=4,
        n_head=2, head_dim=8, backend=backend,
    )
    prompt = tensor_from_numpy(prompt_np, backend=backend)
    prefill_logits = model.prefill(prompt, seq_id=0, block_manager=bm)
    last_tok = int(prefill_logits.to_numpy()[0, -1, :].argmax())
    paged_tokens.append(last_tok)

    for _ in range(3):
        tok   = tensor_from_numpy(np.array([[last_tok]], dtype=np.int32), backend=backend)
        logit = model.decode_step(tok, seq_id=0, block_manager=bm)
        next_tok = int(logit.to_numpy()[0, 0, :].argmax())
        paged_tokens.append(next_tok)
        last_tok = next_tok

    bm.free_seq(0)

    assert paged_tokens == non_paged_tokens, (
        f"Paged {paged_tokens} != non-paged {non_paged_tokens}"
    )


def test_paged_prefill_flash_matches_standard():
    from minitorch.tensor_functions import tensor_from_numpy
    from minitorch.paged_attention import BlockManager

    backend = simple_backend()
    prompt_np = np.array([[4, 9, 2, 7, 1, 3]], dtype=np.int32)

    std_model = _small_model(backend, use_flash_attn=False)
    fa2_model = _small_model(backend, use_flash_attn=True)

    for std_p, fa2_p in zip(std_model.parameters(), fa2_model.parameters()):
        fa2_p.update(std_p.value.detach())

    std_bm = BlockManager(
        num_layers=4, num_blocks=32, block_size=4,
        n_head=2, head_dim=8, backend=backend,
    )
    fa2_bm = BlockManager(
        num_layers=4, num_blocks=32, block_size=4,
        n_head=2, head_dim=8, backend=backend,
    )

    prompt = tensor_from_numpy(prompt_np, backend=backend)

    std_prefill = std_model.prefill(prompt, seq_id=0, block_manager=std_bm)
    fa2_prefill = fa2_model.prefill(prompt, seq_id=0, block_manager=fa2_bm)

    np.testing.assert_allclose(
        fa2_prefill.to_numpy(),
        std_prefill.to_numpy(),
        atol=1e-4,
        rtol=1e-4,
        err_msg="Paged prefill with flash attention does not match standard paged prefill",
    )

    next_tok_std = int(std_prefill.to_numpy()[0, -1, :].argmax())
    next_tok_fa2 = int(fa2_prefill.to_numpy()[0, -1, :].argmax())
    assert next_tok_std == next_tok_fa2

    std_decode_in = tensor_from_numpy(np.array([[next_tok_std]], dtype=np.int32), backend=backend)
    fa2_decode_in = tensor_from_numpy(np.array([[next_tok_fa2]], dtype=np.int32), backend=backend)

    std_decode = std_model.decode_step(std_decode_in, seq_id=0, block_manager=std_bm)
    fa2_decode = fa2_model.decode_step(fa2_decode_in, seq_id=0, block_manager=fa2_bm)

    np.testing.assert_allclose(
        fa2_decode.to_numpy(),
        std_decode.to_numpy(),
        atol=1e-4,
        rtol=1e-4,
        err_msg="Paged decode after flash-based prefill does not match standard paged decode",
    )

    std_bm.free_seq(0)
    fa2_bm.free_seq(0)
