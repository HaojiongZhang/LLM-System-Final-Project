"""Scheduler tests for paged KV-cache admission and preemption."""

import pytest
import numpy as np


def simple_backend():
    import minitorch
    try:
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    except Exception:
        return minitorch.TensorBackend(minitorch.SimpleOps)


def _small_model(backend):
    from minitorch.transformer import DecoderLM
    return DecoderLM(
        n_vocab=16, n_embd=8, n_head=2,
        n_positions=32, p_dropout=0.0,
        backend=backend,
    )


def _make_bm(backend, num_blocks=16, block_size=4):
    from minitorch.paged_attention import BlockManager
    return BlockManager(
        num_layers=4, num_blocks=num_blocks, block_size=block_size,
        n_head=2, head_dim=4, backend=backend,
    )


def test_single_request():
    from minitorch.scheduler import RequestScheduler

    backend = simple_backend()
    model   = _small_model(backend)
    bm      = _make_bm(backend)

    sched = RequestScheduler(model, bm, eos_token_id=999, max_batch_size=4)
    sched.submit(seq_id=0, prompt_tokens=[1, 2, 3])

    outputs = sched.run_until_done(max_new_tokens=5)

    assert 0 in outputs, "Request 0 should complete"
    assert isinstance(outputs[0], list)
    assert len(outputs[0]) <= 5


def test_two_requests_concurrent():
    from minitorch.scheduler import RequestScheduler

    backend = simple_backend()
    model   = _small_model(backend)
    bm      = _make_bm(backend, num_blocks=32)

    sched = RequestScheduler(model, bm, eos_token_id=999, max_batch_size=4)
    sched.submit(seq_id=0, prompt_tokens=[1, 2])
    sched.submit(seq_id=1, prompt_tokens=[3, 4])

    outputs = sched.run_until_done(max_new_tokens=4)

    assert 0 in outputs and 1 in outputs, "Both requests should complete"


def test_preemption_fires():
    from minitorch.scheduler import RequestScheduler

    backend = simple_backend()
    model   = _small_model(backend)
    bm = _make_bm(backend, num_blocks=3, block_size=4)

    sched = RequestScheduler(model, bm, eos_token_id=999, max_batch_size=8)
    sched.submit(seq_id=0, prompt_tokens=[1, 2])
    sched.submit(seq_id=1, prompt_tokens=[3, 4])
    sched.submit(seq_id=2, prompt_tokens=[5, 6, 7, 8, 9])

    outputs = sched.run_until_done(max_new_tokens=6)

    assert set(outputs.keys()) == {0, 1, 2}, (
        f"Expected all 3 requests to complete, got {set(outputs.keys())}"
    )


def test_preempted_output_matches_solo():
    from minitorch.scheduler import RequestScheduler

    backend = simple_backend()
    model   = _small_model(backend)

    bm_solo = _make_bm(backend, num_blocks=32)
    sched_solo = RequestScheduler(model, bm_solo, eos_token_id=999, max_batch_size=4)
    sched_solo.submit(seq_id=0, prompt_tokens=[7, 8, 9])
    solo_out = sched_solo.run_until_done(max_new_tokens=4)

    bm_tight = _make_bm(backend, num_blocks=2, block_size=4)
    sched_tight = RequestScheduler(model, bm_tight, eos_token_id=999, max_batch_size=8)
    sched_tight.submit(seq_id=0, prompt_tokens=[7, 8, 9])
    sched_tight.submit(seq_id=1, prompt_tokens=[1, 2, 3, 4, 5])
    tight_out = sched_tight.run_until_done(max_new_tokens=4)

    assert 0 in tight_out, "Request 0 should complete even after preemption"
    assert tight_out[0] == solo_out[0], (
        f"Preempted output {tight_out[0]} != solo output {solo_out[0]}"
    )


def test_decode_growth_preemption():
    from minitorch.scheduler import RequestScheduler

    backend = simple_backend()
    model   = _small_model(backend)
    bm      = _make_bm(backend, num_blocks=3, block_size=4)

    sched = RequestScheduler(model, bm, eos_token_id=999, max_batch_size=8)
    sched.submit(seq_id=0, prompt_tokens=[1, 2, 3, 4])
    sched.submit(seq_id=1, prompt_tokens=[5, 6, 7, 8])

    outputs = sched.run_until_done(max_new_tokens=2)
    assert set(outputs.keys()) == {0, 1}
