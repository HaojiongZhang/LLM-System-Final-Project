import argparse
import time

import numpy as np

from minitorch.paged_attention import BlockManager


def run_trial(num_sequences, sequence_length, block_size, n_head, head_dim):
    blocks_per_seq = (sequence_length + block_size - 1) // block_size
    manager = BlockManager(
        num_layers=1,
        num_blocks=num_sequences * blocks_per_seq,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
        backend=None,
    )

    k_vec = np.ones((n_head, head_dim), dtype=np.float32)
    v_vec = np.ones((n_head, head_dim), dtype=np.float32)

    start = time.perf_counter()
    for seq_id in range(num_sequences):
        manager.allocate_seq(seq_id)
        for pos in range(sequence_length):
            manager.write_kv(0, seq_id, pos, k_vec, v_vec)
        manager.gather_kv(0, seq_id, sequence_length)
    elapsed = time.perf_counter() - start

    tokens = num_sequences * sequence_length
    return tokens / elapsed, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark paged KV-cache writes and gathers.")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    args = parser.parse_args()

    tokens_per_second, elapsed = run_trial(
        args.num_sequences,
        args.sequence_length,
        args.block_size,
        args.n_head,
        args.head_dim,
    )
    print(f"elapsed_s={elapsed:.4f}")
    print(f"tokens_per_second={tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
