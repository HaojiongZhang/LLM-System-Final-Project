import numpy as np


def simple_backend():
    import minitorch

    try:
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    except Exception:
        return minitorch.TensorBackend(minitorch.SimpleOps)


def _make_models(backend):
    from minitorch.transformer import DecoderLM

    base = DecoderLM(
        n_vocab=64,
        n_embd=32,
        n_head=4,
        n_positions=64,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=False,
    )
    flash = DecoderLM(
        n_vocab=64,
        n_embd=32,
        n_head=4,
        n_positions=64,
        p_dropout=0.0,
        backend=backend,
        use_flash_attn=True,
    )
    for base_p, flash_p in zip(base.parameters(), flash.parameters()):
        flash_p.update(base_p.value.detach())
    base.eval()
    flash.eval()
    return base, flash


def _run_recompute(model, prompt_np, new_tokens, backend):
    from minitorch.tensor_functions import tensor_from_numpy

    ctx_np = prompt_np.copy()
    tokens = []
    logits_steps = []
    for _ in range(new_tokens):
        logits = model(tensor_from_numpy(ctx_np, backend=backend))
        last_logits = np.ascontiguousarray(logits.to_numpy()[:, -1, :], dtype=np.float32)
        logits_steps.append(last_logits.copy())
        next_tok = last_logits.argmax(axis=-1).astype(np.int32)
        tokens.append(next_tok.copy())
        ctx_np = np.concatenate([ctx_np, next_tok[:, None]], axis=1)
    return np.stack(tokens, axis=1), logits_steps


def _run_paged(model, prompt_np, new_tokens, backend):
    from minitorch.paged_attention import BlockManager
    from minitorch.tensor_functions import tensor_from_numpy

    batch_size, prompt_len = prompt_np.shape
    block_manager = BlockManager(
        num_layers=4,
        num_blocks=batch_size * 32,
        block_size=4,
        n_head=4,
        head_dim=8,
        backend=backend,
    )
    seq_ids = list(range(batch_size))
    first_tokens = []
    logits_steps = []
    for sid in seq_ids:
        logits = model.prefill(
            tensor_from_numpy(prompt_np[sid : sid + 1], backend=backend),
            seq_id=sid,
            block_manager=block_manager,
        )
        last_logits = np.ascontiguousarray(logits.to_numpy()[0, -1, :], dtype=np.float32)
        logits_steps.append(last_logits.copy())
        first_tokens.append(int(last_logits.argmax()))
    token_ids_np = np.array(first_tokens, dtype=np.int32)[:, None]
    logits_steps = [np.stack(logits_steps, axis=0)]
    tokens = [token_ids_np[:, 0].copy()]
    for _ in range(new_tokens - 1):
        logits = model.decode_step_batch(
            tensor_from_numpy(token_ids_np, backend=backend),
            seq_ids,
            block_manager,
        )
        last_logits = np.ascontiguousarray(logits.to_numpy()[:, 0, :], dtype=np.float32)
        logits_steps.append(last_logits.copy())
        token_ids_np = last_logits.argmax(axis=-1).astype(np.int32)[:, None]
        tokens.append(token_ids_np[:, 0].copy())
    return np.stack(tokens, axis=1), logits_steps


def test_flash_recompute_matches_dense_recompute():
    backend = simple_backend()
    dense, flash = _make_models(backend)
    prompt_np = np.array([[4, 9, 2, 7, 1, 3]], dtype=np.int32)

    dense_tokens, dense_logits = _run_recompute(dense, prompt_np, new_tokens=4, backend=backend)
    flash_tokens, flash_logits = _run_recompute(flash, prompt_np, new_tokens=4, backend=backend)

    np.testing.assert_array_equal(flash_tokens, dense_tokens)
    for flash_step, dense_step in zip(flash_logits, dense_logits):
        np.testing.assert_allclose(flash_step, dense_step, atol=1e-4, rtol=1e-4)


def test_paged_modes_match_dense_reference():
    backend = simple_backend()
    dense, flash = _make_models(backend)
    prompt_np = np.array([[5, 6, 7, 8]], dtype=np.int32)

    dense_tokens, dense_logits = _run_recompute(dense, prompt_np, new_tokens=4, backend=backend)
    dense_paged_tokens, dense_paged_logits = _run_paged(dense, prompt_np, new_tokens=4, backend=backend)
    flash_paged_tokens, flash_paged_logits = _run_paged(flash, prompt_np, new_tokens=4, backend=backend)

    np.testing.assert_array_equal(dense_paged_tokens, dense_tokens)
    np.testing.assert_array_equal(flash_paged_tokens, dense_tokens)
    for dense_paged_step, dense_step in zip(dense_paged_logits, dense_logits):
        np.testing.assert_allclose(dense_paged_step, dense_step, atol=1e-4, rtol=1e-4)
    for flash_paged_step, dense_step in zip(flash_paged_logits, dense_logits):
        np.testing.assert_allclose(flash_paged_step, dense_step, atol=1e-4, rtol=1e-4)
