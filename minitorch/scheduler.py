"""
RequestScheduler for FIFO admission and preemption with a paged KV cache.

Sits above BlockManager and DecoderLM.  Manages which sequences are admitted
into the active decode pool at each step and handles block-pool pressure by
preempting running sequences when a new request cannot be admitted.

Admission policy:
    A pending request of prompt length L requires ceil(L / block_size) blocks
    before prefill begins.  If the pool is too full the scheduler preempts the
    *lowest-priority* running sequence (longest running = most blocks consumed),
    returns its blocks, re-queues it, and retries admission.

Decode loop:
    Each call to step() performs:
      1. Admit as many pending requests as the pool allows.
      2. Batch-decode one new token for every active sequence.
      3. Detect EOS / max-length completions, free their blocks, emit output.

Usage::

    scheduler = RequestScheduler(model, block_manager, eos_token_id=50256)
    for seq_id, tokens in enumerate(prompts):
        scheduler.submit(seq_id, tokens)
    outputs = scheduler.run_until_done(max_new_tokens=50)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class RequestScheduler:
    """
    FIFO scheduler with preemption over a shared BlockManager pool.

    Args:
        model:           DecoderLM instance.
        block_manager:   BlockManager instance (shared pool).
        eos_token_id:    Token id that signals end-of-sequence.
        max_batch_size:  Cap on simultaneous active sequences.
    """

    def __init__(
        self,
        model,
        block_manager,
        eos_token_id: int = 0,
        max_batch_size: int = 32,
    ):
        self.model           = model
        self.bm              = block_manager
        self.eos_token_id    = eos_token_id
        self.max_batch_size  = max_batch_size

        # Pending requests: deque of (seq_id, prompt_token_ids)
        self._pending: deque = deque()

        # Active sequences: seq_id -> list of generated token ids so far
        self._running: Dict[int, List[int]] = {}

        # Completed outputs: seq_id -> full token list (prompt + generated)
        self._completed: Dict[int, List[int]] = {}

        # Original prompts (used to slice generated output from full token history).
        self._prompts: Dict[int, List[int]] = {}

        # Decode-step counts used to enforce max_new_tokens across run_until_done().
        self._step_counts: Dict[int, int] = {}

        # Sequence ids currently waiting after being preempted. These requests
        # may be re-admitted when capacity becomes naturally available, but they
        # should not trigger further preemption of other running sequences or
        # the scheduler can thrash indefinitely.
        self._preempted_waiting: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, seq_id: int, prompt_tokens: List[int]) -> None:
        """Enqueue a new generation request."""
        self._prompts[seq_id] = list(prompt_tokens)
        self._step_counts[seq_id] = 0
        self._pending.append((seq_id, list(prompt_tokens)))

    def step(self) -> Dict[int, List[int]]:
        """
        One decode tick.

        Returns a dict of {seq_id: token_list} for sequences that completed
        during this step (may be empty).
        """
        self._admit_pending()

        if not self._running:
            return {}

        self._ensure_decode_capacity()

        if not self._running:
            return {}

        completed_this_step = self._decode_and_collect()
        return completed_this_step

    def run_until_done(self, max_new_tokens: int = 128) -> Dict[int, List[int]]:
        """
        Loop until all submitted requests are completed or hit max_new_tokens.

        Returns:
            dict mapping seq_id -> generated token ids (excluding prompt).
        """
        idle_loops = 0
        while self._pending or self._running:
            before_pending = len(self._pending)
            before_running = len(self._running)
            before_completed = len(self._completed)
            before_steps = dict(self._step_counts)

            completed = self.step()
            for sid, toks in completed.items():
                self._completed[sid] = toks
            for sid in list(self._running.keys()):
                self._step_counts[sid] = self._step_counts.get(sid, 0) + 1
                if self._step_counts[sid] >= max_new_tokens:
                    self._force_complete(sid)

            progress = (
                len(self._pending) != before_pending
                or len(self._running) != before_running
                or len(self._completed) != before_completed
                or any(
                    self._step_counts.get(sid, 0) != before_steps.get(sid, 0)
                    for sid in set(before_steps) | set(self._step_counts)
                )
            )
            if progress:
                idle_loops = 0
            else:
                idle_loops += 1
                if idle_loops >= 8:
                    raise RuntimeError(
                        "RequestScheduler made no progress for multiple iterations. "
                        f"pending={list(self._pending)} running={list(self._running.keys())} "
                        f"free_blocks={len(self.bm.free_blocks)} "
                        f"seq_lengths={dict(self.bm.seq_lengths)}"
                    )

        return {
            sid: self._completed[sid]
            for sid in sorted(self._completed.keys())
        }

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def _blocks_needed(self, prompt_len: int) -> int:
        return math.ceil(prompt_len / self.bm.block_size)

    def _admit_pending(self) -> None:
        """Greedily admit pending requests that fit in the pool."""
        # Simple anti-thrashing policy for this project: once any sequence has
        # been preempted, let the current running set drain before trying to
        # re-admit queued work.
        if self._preempted_waiting and self._running:
            return

        while (
            self._pending
            and len(self._running) < self.max_batch_size
        ):
            seq_id, context_tokens = self._pending[0]
            needed = self._blocks_needed(len(context_tokens))

            if needed > self.bm.num_blocks:
                raise RuntimeError(
                    f"seq_id {seq_id} prompt needs {needed} KV blocks, "
                    f"but pool capacity is only {self.bm.num_blocks}"
                )

            if len(self.bm.free_blocks) >= needed:
                self._pending.popleft()
                self._preempted_waiting.discard(seq_id)
                self._prefill(seq_id, context_tokens)
            else:
                if seq_id in self._preempted_waiting:
                    break
                # Try to free space by preempting the lowest-priority sequence.
                if not self._maybe_preempt(needed):
                    break  # Pool is full and nothing can be preempted safely.

    def _maybe_preempt(self, needed_blocks: int) -> bool:
        """
        Preempt the running sequence that has consumed the most blocks
        (longest history = lowest priority for continued running).

        Returns True if a preemption occurred, False if nothing could be freed.
        """
        if not self._running:
            return False

        # Pick the sequence with the largest committed length (most blocks).
        victim_id = max(
            self._running.keys(),
            key=lambda sid: self.bm.seq_lengths.get(sid, 0),
        )

        # Re-queue victim with its full token history so a later prefill
        # reconstructs the exact KV state before decoding resumes.
        #
        # Important: append to the back of the pending queue. If we appendleft,
        # the scheduler can livelock by immediately re-admitting the victim it
        # just evicted, leaving the blocked larger request at the front forever.
        victim_tokens = list(self._running[victim_id])
        self.bm.free_seq(victim_id)
        del self._running[victim_id]
        self._preempted_waiting.add(victim_id)
        self._pending.append((victim_id, victim_tokens))
        return True

    def _decode_extra_blocks_needed(self) -> int:
        """
        Count how many running sequences will cross a block boundary on the next
        decode token and therefore need one additional KV block.
        """
        needed = 0
        for sid in self._running.keys():
            cur_len = self.bm.seq_lengths.get(sid, 0)
            next_total = cur_len + 1
            if self._blocks_needed(next_total) > self.bm.num_blocks:
                raise RuntimeError(
                    f"seq_id {sid} would require {self._blocks_needed(next_total)} "
                    f"KV blocks after the next decode step, exceeding pool "
                    f"capacity {self.bm.num_blocks}"
                )
            if cur_len > 0 and cur_len % self.bm.block_size == 0:
                needed += 1
        return needed

    def _ensure_decode_capacity(self) -> None:
        """
        Preempt running sequences until the next batched decode step has enough
        free KV blocks for all sequences that need to grow their cache.
        """
        while self._running:
            needed = self._decode_extra_blocks_needed()
            if len(self.bm.free_blocks) >= needed:
                return
            if not self._maybe_preempt(needed):
                break

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill(self, seq_id: int, context_tokens: List[int]) -> None:
        """Run prefill for a sequence context and register it as active."""
        from .tensor_functions import tensor_from_numpy

        prompt_np = np.array([context_tokens], dtype=np.int32)
        idx = tensor_from_numpy(prompt_np, backend=self.model.backend)

        self.model.prefill(idx, seq_id, self.bm)
        self._running[seq_id] = list(context_tokens)

    # ------------------------------------------------------------------
    # Decode tick
    # ------------------------------------------------------------------

    def _decode_and_collect(self) -> Dict[int, List[int]]:
        """
        One batched decode step across all active sequences.
        Returns completed {seq_id: generated_tokens} for this step.
        """
        from .tensor_functions import tensor_from_numpy

        seq_ids   = list(self._running.keys())
        B         = len(seq_ids)
        # Build (B, 1) token tensor: last token generated per sequence.
        last_tokens = np.array(
            [[self._running[sid][-1]] for sid in seq_ids], dtype=np.int32
        )
        token_ids = tensor_from_numpy(last_tokens, backend=self.model.backend)

        logits = self.model.decode_step_batch(token_ids, seq_ids, self.bm)
        # logits shape: (B, 1, n_vocab)
        logits_np = logits.to_numpy()  # (B, 1, n_vocab)
        next_tokens = logits_np[:, 0, :].argmax(axis=-1).astype(np.int32)  # (B,)

        completed = {}
        for i, sid in enumerate(seq_ids):
            tok = int(next_tokens[i])
            self._running[sid].append(tok)
            if tok == self.eos_token_id:
                generated = self._running[sid][len(self._prompts[sid]):]
                completed[sid] = generated
                self.bm.free_seq(sid)
                del self._running[sid]
                self._step_counts.pop(sid, None)
                self._preempted_waiting.discard(sid)

        return completed

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _force_complete(self, seq_id: int) -> None:
        """Mark a sequence as completed due to hitting max_new_tokens."""
        if seq_id in self._running:
            generated = self._running[seq_id][len(self._prompts[seq_id]):]
            self._completed[seq_id] = generated
            self.bm.free_seq(seq_id)
            del self._running[seq_id]
        self._step_counts.pop(seq_id, None)
        self._preempted_waiting.discard(seq_id)

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    @property
    def n_running(self) -> int:
        return len(self._running)
