"""Paged KV-cache block management for inference."""

import numpy as np
from typing import Dict, List, Tuple


class BlockManager:
    """Shared KV-cache block pool."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        n_head: int,
        head_dim: int,
        backend,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = head_dim
        self.backend = backend

        # K and V are stored separately. All layers share the same block pool.
        self.kv_k = np.zeros(
            (num_layers, num_blocks, n_head, block_size, head_dim),
            dtype=np.float32,
        )
        self.kv_v = np.zeros(
            (num_layers, num_blocks, n_head, block_size, head_dim),
            dtype=np.float32,
        )

        self.free_blocks: List[int] = list(range(num_blocks))

        self.block_tables: Dict[int, List[int]] = {}
        # Tokens committed so far for each active sequence.
        self.seq_lengths: Dict[int, int] = {}

    def allocate_seq(self, seq_id: int) -> None:
        """Register a new sequence and allocate its first block."""
        if seq_id in self.block_tables:
            raise ValueError(f"seq_id {seq_id} is already active; call free_seq first")
        self.block_tables[seq_id] = [self._alloc_block()]
        self.seq_lengths[seq_id] = 0

    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        token_pos: int,
        k_vec: np.ndarray,
        v_vec: np.ndarray,
    ) -> None:
        """Write K and V vectors for one token position into the cache."""
        block_idx = token_pos // self.block_size
        slot_idx = token_pos % self.block_size

        # Multiple layers write the same token position, so the block may exist.
        while block_idx >= len(self.block_tables[seq_id]):
            self.block_tables[seq_id].append(self._alloc_block())

        phys_block = self.block_tables[seq_id][block_idx]
        self.kv_k[layer_idx, phys_block, :, slot_idx, :] = k_vec
        self.kv_v[layer_idx, phys_block, :, slot_idx, :] = v_vec

    def gather_kv(
        self,
        layer_idx: int,
        seq_id: int,
        count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Copy the first count K/V vectors for a sequence into contiguous arrays."""
        k_out = np.empty((1, self.n_head, count, self.head_dim), dtype=np.float32)
        v_out = np.empty((1, self.n_head, count, self.head_dim), dtype=np.float32)

        for i, phys_block in enumerate(self.block_tables[seq_id]):
            src_start = i * self.block_size
            src_end   = min(src_start + self.block_size, count)
            if src_start >= count:
                break
            n_slots = src_end - src_start
            k_out[0, :, src_start:src_end, :] = self.kv_k[layer_idx, phys_block, :, :n_slots, :]
            v_out[0, :, src_start:src_end, :] = self.kv_v[layer_idx, phys_block, :, :n_slots, :]

        return k_out, v_out

    def gather_kv_padded(
        self,
        layer_idx: int,
        seq_ids: List[int],
        counts: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch version of gather_kv with zero padding to the max sequence length."""
        B = len(seq_ids)
        max_count = max(counts)

        k_out = np.zeros((B, self.n_head, max_count, self.head_dim), dtype=np.float32)
        v_out = np.zeros((B, self.n_head, max_count, self.head_dim), dtype=np.float32)

        for b, (seq_id, count) in enumerate(zip(seq_ids, counts)):
            k_b, v_b = self.gather_kv(layer_idx, seq_id, count)
            k_out[b, :, :count, :] = k_b[0]
            v_out[b, :, :count, :] = v_b[0]

        return k_out, v_out

    def get_block_tables_np(self, seq_ids: List[int]) -> np.ndarray:
        """Pack block tables into a contiguous int32 array for the C kernel."""
        B = len(seq_ids)
        max_blocks = max(len(self.block_tables[sid]) for sid in seq_ids)
        out = np.full((B, max_blocks), -1, dtype=np.int32)
        for b, sid in enumerate(seq_ids):
            table = self.block_tables[sid]
            out[b, : len(table)] = table
        return out

    def free_seq(self, seq_id: int) -> None:
        """Return all blocks for a finished sequence to the free pool."""
        for phys_block in self.block_tables[seq_id]:
            self.free_blocks.append(phys_block)
        del self.block_tables[seq_id]
        del self.seq_lengths[seq_id]

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError(
                "KV cache pool is full; increase num_blocks or reduce max sequence length"
            )
        return self.free_blocks.pop()
