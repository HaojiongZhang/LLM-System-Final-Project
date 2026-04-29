"""
Benchmark: FIFO+preemption scheduler vs. static (no-preemption) scheduler.

Metrics measured:
  1. Admission latency (time-to-first-token, TTFT) per request.
  2. Block utilization over time (fraction of pool blocks in use).
  3. Preemption count for the FIFO+preemption scheduler.
  4. End-to-end throughput (tokens/sec) for both schedulers.

The static scheduler pre-reserves ceil(max_prompt_len / block_size) blocks for
each sequence it admits.  It does not preempt and can therefore only run at most
floor(pool_size / reserved_blocks_per_seq) sequences concurrently.  Requests
beyond that cap queue until a slot frees, regardless of their actual length.

Usage:
    # Fast local run (small model, short prompts):
    python scripts/benchmark_scheduler.py

    # HPC run with realistic sizes:
    python scripts/benchmark_scheduler.py --n-requests 16 --min-len 256 --max-len 2048 \
        --n-embd 256 --n-head 8 --num-blocks 512 --block-size 16

Run on CPU (SimpleOps) by default.  Pass --backend cuda to use CudaKernelOps.
"""

import argparse
import math
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import sys

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Static scheduler (baseline, no preemption)
# ---------------------------------------------------------------------------

class StaticScheduler:
    """
    Baseline scheduler that reserves a fixed block budget per sequence.

    Admission policy:
        Each sequence reserves `blocks_per_slot = ceil(max_seq_len / block_size)`
        blocks regardless of its actual prompt length.  A new sequence is admitted
        only when `free_slots > 0`.  No preemption.

    This is what a naive implementation would do; it avoids fragmentation
    concerns at the cost of severely under-utilising the block pool whenever
    requests are shorter than max_seq_len.
    """

    def __init__(self, model, block_manager, eos_token_id=0,
                 max_batch_size=32, max_seq_len=64):
        self.model           = model
        self.bm              = block_manager
        self.eos_token_id    = eos_token_id
        self.max_batch_size  = max_batch_size
        # How many blocks each slot pre-reserves (fixed, pessimistic).
        self.blocks_per_slot = math.ceil(max_seq_len / block_manager.block_size)

        self._pending: deque          = deque()
        self._running: Dict[int, List[int]] = {}
        self._completed: Dict[int, List[int]] = {}
        self._prompts: Dict[int, List[int]] = {}
        # Track reserved-slot count (not actual block_tables).
        self._reserved_slots: int = 0

    def submit(self, seq_id: int, prompt_tokens: List[int]) -> None:
        self._prompts[seq_id] = list(prompt_tokens)
        self._pending.append((seq_id, list(prompt_tokens)))

    def _max_concurrent(self) -> int:
        return min(
            self.max_batch_size,
            len(self.bm.free_blocks) // max(self.blocks_per_slot, 1),
        )

    def _admit_pending(self) -> None:
        while self._pending and len(self._running) < self._max_concurrent():
            seq_id, prompt_tokens = self._pending.popleft()
            self._reserved_slots += self.blocks_per_slot
            self._prefill(seq_id, prompt_tokens)

    def _prefill(self, seq_id: int, prompt_tokens: List[int]) -> None:
        from minitorch.tensor_functions import tensor_from_numpy
        prompt_np = np.array([prompt_tokens], dtype=np.int32)
        idx = tensor_from_numpy(prompt_np, backend=self.model.backend)
        self.model.prefill(idx, seq_id, self.bm)
        self._running[seq_id] = list(prompt_tokens)

    def _decode_and_collect(self) -> Dict[int, List[int]]:
        from minitorch.tensor_functions import tensor_from_numpy
        if not self._running:
            return {}
        seq_ids    = list(self._running.keys())
        last_toks  = np.array([[self._running[s][-1]] for s in seq_ids], dtype=np.int32)
        token_ids  = tensor_from_numpy(last_toks, backend=self.model.backend)
        logits     = self.model.decode_step_batch(token_ids, seq_ids, self.bm)
        logits_np  = logits.to_numpy()
        next_toks  = logits_np[:, 0, :].argmax(axis=-1).astype(np.int32)

        completed = {}
        for i, sid in enumerate(seq_ids):
            tok = int(next_toks[i])
            self._running[sid].append(tok)
            if tok == self.eos_token_id:
                generated = self._running[sid][len(self._prompts[sid]):]
                completed[sid] = generated
                self.bm.free_seq(sid)
                del self._running[sid]
                self._reserved_slots -= self.blocks_per_slot
        return completed

    def step(self) -> Dict[int, List[int]]:
        self._admit_pending()
        return self._decode_and_collect()

    def run_until_done(self, max_new_tokens: int = 32) -> Dict[int, List[int]]:
        step_counts: Dict[int, int] = {}
        while self._pending or self._running:
            completed = self.step()
            for sid, toks in completed.items():
                self._completed[sid] = toks
            for sid in list(self._running.keys()):
                step_counts[sid] = step_counts.get(sid, 0) + 1
                if step_counts[sid] >= max_new_tokens:
                    if sid in self._running:
                        generated = self._running[sid][len(self._prompts[sid]):]
                        self._completed[sid] = generated
                        self.bm.free_seq(sid)
                        del self._running[sid]
                        self._reserved_slots -= self.blocks_per_slot
        return {sid: self._completed[sid] for sid in sorted(self._completed)}


# ---------------------------------------------------------------------------
# Instrumented RequestScheduler (wraps minitorch.scheduler.RequestScheduler)
# ---------------------------------------------------------------------------

class InstrumentedScheduler:
    """
    Thin wrapper around RequestScheduler that records TTFT, preemption count,
    and per-step block utilisation.
    """

    def __init__(self, model, block_manager, eos_token_id=0, max_batch_size=32):
        from minitorch.scheduler import RequestScheduler
        self._sched = RequestScheduler(
            model, block_manager,
            eos_token_id=eos_token_id,
            max_batch_size=max_batch_size,
        )
        self.bm               = block_manager
        self._total_blocks    = len(block_manager.free_blocks) + 0  # will grow as alloc
        self._submit_times: Dict[int, float]  = {}
        self._ttft:         Dict[int, float]  = {}
        self._first_seen:   set               = set()
        self.preemption_count: int            = 0
        self.utilisation_log: List[float]     = []  # fraction at each step
        self._prompts: Dict[int, List[int]]   = {}

        # Monkey-patch preemption to count it.
        orig_preempt = self._sched._maybe_preempt
        counter = [0]

        def _counted_preempt(needed):
            result = orig_preempt(needed)
            if result:
                counter[0] += 1
            return result

        self._sched._maybe_preempt = _counted_preempt
        self._counter = counter

    @property
    def n_pending(self) -> int:
        return self._sched.n_pending

    @property
    def n_running(self) -> int:
        return self._sched.n_running

    def submit(self, seq_id: int, prompt_tokens: List[int]) -> None:
        self._submit_times[seq_id] = time.perf_counter()
        self._prompts[seq_id] = list(prompt_tokens)
        self._sched.submit(seq_id, prompt_tokens)

    def run_until_done(self, max_new_tokens: int = 32) -> Dict[int, List[int]]:
        step_counts: Dict[int, int] = {}
        total_blocks = len(self.bm.free_blocks)  # snapshot before any alloc

        while self._sched._pending or self._sched._running:
            self._sched._admit_pending()

            # Record TTFT for newly-admitted sequences (first time they appear
            # in _running but haven't been seen before).
            now = time.perf_counter()
            for sid in self._sched._running:
                if sid not in self._first_seen:
                    self._first_seen.add(sid)
                    self._ttft[sid] = now - self._submit_times.get(sid, now)

            if not self._sched._running:
                break

            completed = self._sched._decode_and_collect()
            for sid, toks in completed.items():
                self._sched._completed[sid] = toks

            for sid in list(self._sched._running.keys()):
                step_counts[sid] = step_counts.get(sid, 0) + 1
                if step_counts[sid] >= max_new_tokens:
                    self._sched._force_complete(sid)

            # Log utilisation: fraction of pool blocks NOT free.
            free = len(self.bm.free_blocks)
            used = total_blocks - free
            self.utilisation_log.append(used / max(total_blocks, 1))

        self.preemption_count = self._counter[0]
        return {
            sid: self._sched._completed[sid]
            for sid in sorted(self._sched._completed)
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _make_model(backend, n_vocab, n_embd, n_head, n_positions):
    from minitorch.transformer import DecoderLM
    return DecoderLM(
        n_vocab=n_vocab, n_embd=n_embd, n_head=n_head,
        n_positions=n_positions, p_dropout=0.0, backend=backend,
    )


def _make_bm(backend, num_layers, num_blocks, block_size, n_head, head_dim):
    from minitorch.paged_attention import BlockManager
    return BlockManager(
        num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
        n_head=n_head, head_dim=head_dim, backend=backend,
    )


def _generate_workload(n_requests, min_len, max_len, vocab_size, seed=0):
    rng = np.random.default_rng(seed)
    lengths  = rng.integers(min_len, max_len + 1, size=n_requests)
    prompts  = [rng.integers(1, vocab_size, size=int(l), dtype=np.int32).tolist()
                for l in lengths]
    return prompts


def run_benchmark(args):
    import minitorch

    if args.backend == "cuda":
        backend = minitorch.TensorBackend(minitorch.CudaKernelOps)
    else:
        backend = minitorch.TensorBackend(minitorch.SimpleOps)

    head_dim     = args.n_embd // args.n_head
    n_layers     = 2  # keep shallow for speed; real use: match model depth
    n_vocab      = args.n_vocab

    model        = _make_model(backend, n_vocab, args.n_embd, args.n_head, args.n_positions)
    prompts      = _generate_workload(args.n_requests, args.min_len, args.max_len, n_vocab)

    print(f"\nWorkload: {args.n_requests} requests, "
          f"prompt lengths U[{args.min_len}, {args.max_len}], "
          f"backend={args.backend}")
    print(f"Model: n_embd={args.n_embd}, n_head={args.n_head}, "
          f"n_vocab={n_vocab}, n_layers={n_layers}")
    print(f"Pool: num_blocks={args.num_blocks}, block_size={args.block_size}")

    # ---------- FIFO+preemption scheduler ----------
    bm_fifo = _make_bm(backend, n_layers, args.num_blocks, args.block_size,
                       args.n_head, head_dim)
    isched  = InstrumentedScheduler(
        model, bm_fifo,
        eos_token_id=args.eos_token_id,
        max_batch_size=args.max_batch_size,
    )
    for i, p in enumerate(prompts):
        isched.submit(i, p)

    t0_fifo = time.perf_counter()
    fifo_out = isched.run_until_done(max_new_tokens=args.max_new_tokens)
    t1_fifo  = time.perf_counter()
    fifo_wall = t1_fifo - t0_fifo

    total_fifo_tokens = sum(len(toks) for toks in fifo_out.values())
    fifo_throughput   = total_fifo_tokens / fifo_wall if fifo_wall > 0 else 0.0

    ttft_vals = list(isched._ttft.values())
    mean_ttft = np.mean(ttft_vals) * 1000 if ttft_vals else float("nan")  # ms
    p50_ttft  = np.percentile(ttft_vals, 50) * 1000 if ttft_vals else float("nan")
    p99_ttft  = np.percentile(ttft_vals, 99) * 1000 if ttft_vals else float("nan")

    mean_util = np.mean(isched.utilisation_log) if isched.utilisation_log else 0.0
    peak_util = np.max(isched.utilisation_log)  if isched.utilisation_log else 0.0

    print("\n=== FIFO + Preemption Scheduler ===")
    print(f"  Completed requests   : {len(fifo_out)} / {args.n_requests}")
    print(f"  Preemption count     : {isched.preemption_count}")
    print(f"  Wall time            : {fifo_wall:.2f}s")
    print(f"  Throughput           : {fifo_throughput:.1f} tokens/s")
    print(f"  TTFT  mean / p50 / p99 : {mean_ttft:.1f} / {p50_ttft:.1f} / {p99_ttft:.1f} ms")
    print(f"  Block utilisation  mean / peak : {mean_util:.1%} / {peak_util:.1%}")

    # ---------- Static (no-preemption) scheduler ----------
    max_prompt_len = max(len(p) for p in prompts)
    bm_static = _make_bm(backend, n_layers, args.num_blocks, args.block_size,
                         args.n_head, head_dim)
    static_sched = StaticScheduler(
        model, bm_static,
        eos_token_id=args.eos_token_id,
        max_batch_size=args.max_batch_size,
        max_seq_len=max_prompt_len + args.max_new_tokens,
    )
    for i, p in enumerate(prompts):
        static_sched.submit(i, p)

    t0_st = time.perf_counter()
    static_out = static_sched.run_until_done(max_new_tokens=args.max_new_tokens)
    t1_st = time.perf_counter()
    static_wall = t1_st - t0_st

    total_static_tokens = sum(len(toks) for toks in static_out.values())
    static_throughput   = total_static_tokens / static_wall if static_wall > 0 else 0.0

    print("\n=== Static Scheduler (no preemption) ===")
    print(f"  Completed requests   : {len(static_out)} / {args.n_requests}")
    print(f"  blocks_per_slot      : {static_sched.blocks_per_slot}  "
          f"(reserves {static_sched.blocks_per_slot * args.block_size} tokens/seq "
          f"regardless of actual length)")
    concurrent = len(bm_static.free_blocks) // max(static_sched.blocks_per_slot, 1)
    print(f"  Max concurrent seqs  : {min(args.max_batch_size, concurrent)}")
    print(f"  Wall time            : {static_wall:.2f}s")
    print(f"  Throughput           : {static_throughput:.1f} tokens/s")

    # ---------- Summary ----------
    if static_wall > 0 and fifo_wall > 0:
        speedup = static_wall / fifo_wall
    else:
        speedup = float("nan")

    print("\n=== Comparison ===")
    print(
        f"  FIFO wall / Static wall : {fifo_wall:.2f}s / {static_wall:.2f}s  "
        f"FIFO {speedup:.2f}x {'faster' if speedup >= 1.0 else 'slower'}"
    )
    tput_ratio = fifo_throughput / static_throughput if static_throughput > 0 else float("nan")
    print(f"  Throughput ratio (FIFO / Static) : {tput_ratio:.2f}x")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Workload
    parser.add_argument("--n-requests",    type=int, default=8,
                        help="Number of generation requests (default: 8 for local; 16 for HPC)")
    parser.add_argument("--min-len",       type=int, default=4,
                        help="Min prompt length (default: 4; use 256 on HPC)")
    parser.add_argument("--max-len",       type=int, default=12,
                        help="Max prompt length (default: 12; use 2048 on HPC)")
    parser.add_argument("--max-new-tokens",type=int, default=8,
                        help="Max generated tokens per request")
    parser.add_argument("--eos-token-id",  type=int, default=99999,
                        help="EOS token id (default: 99999 = won't fire naturally)")

    # Model
    parser.add_argument("--n-vocab",       type=int, default=64)
    parser.add_argument("--n-embd",        type=int, default=32)
    parser.add_argument("--n-head",        type=int, default=2)
    parser.add_argument("--n-positions",   type=int, default=64)

    # Block pool
    parser.add_argument("--num-blocks",    type=int, default=32,
                        help="Total blocks in pool (default: 32; use 512+ on HPC)")
    parser.add_argument("--block-size",    type=int, default=4)

    # Scheduler
    parser.add_argument("--max-batch-size",type=int, default=16)

    # Backend
    parser.add_argument("--backend",       default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
