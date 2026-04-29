"""
Paged KV cache for inference.

KV cache is divided into fixed-size blocks rather than one large contiguous
buffer per sequence.  A per-sequence block table maps logical token positions
to physical blocks in a shared pool, allowing multiple sequences to share the
pool and releasing blocks back when a sequence finishes.

Physical layout:
    kv_k / kv_v: (num_layers, num_blocks, n_head, block_size, head_dim)
    kv_k[layer_idx, phys_block, head, slot, dim]

Token position T for seq_id maps to:
    phys_block = block_tables[seq_id][T // block_size]
    slot       = T % block_size

Storage:
    Primary:  numpy arrays (kv_k, kv_v) on CPU, always consistent and used for
              the gather-fallback path when paged_attn.so is not compiled.
    GPU copy: pycuda DeviceAllocation mirrors kv_k/kv_v on device for the CUDA
              kernel path.  Updated incrementally on every write_kv call so the
              device copy stays in sync without a full pool transfer per step.
"""

import ctypes
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import pycuda.driver as _cuda_driver
    import pycuda.autoinit  # noqa: F401 - initializes the CUDA context
    _PYCUDA_AVAILABLE = True
except Exception:
    _cuda_driver = None          # type: ignore[assignment]
    _PYCUDA_AVAILABLE = False


class BlockManager:
    """
    Shared KV cache block pool backed by numpy arrays with an optional GPU
    mirror for the CUDA decode kernel.

    Args:
        num_layers:  Transformer layer count.
        num_blocks:  Total blocks in the pool.
        block_size:  Tokens per block.
        n_head:      Attention head count.
        head_dim:    Dimension per head (n_embd // n_head).
        backend:     minitorch TensorBackend.
    """

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
        self.n_head     = n_head
        self.head_dim   = head_dim
        self.backend    = backend

        # Primary storage on CPU (always authoritative).
        self.kv_k = np.zeros(
            (num_layers, num_blocks, n_head, block_size, head_dim),
            dtype=np.float32,
        )
        self.kv_v = np.zeros_like(self.kv_k)

        # Optional GPU mirror for paged_attn.so kernel path.
        self._kv_k_gpu: Optional[object] = None  # pycuda DeviceAllocation
        self._kv_v_gpu: Optional[object] = None

        if _PYCUDA_AVAILABLE:
            nbytes = self.kv_k.nbytes
            self._kv_k_gpu = _cuda_driver.mem_alloc(nbytes)
            self._kv_v_gpu = _cuda_driver.mem_alloc(nbytes)
            # Initialise GPU buffers to zero.
            _cuda_driver.memcpy_htod(self._kv_k_gpu, self.kv_k)
            _cuda_driver.memcpy_htod(self._kv_v_gpu, self.kv_v)

        self.free_blocks: List[int] = list(range(num_blocks))

        self.block_tables: Dict[int, List[int]] = {}
        # seq_lengths[seq_id] = tokens committed so far (same across all layers).
        self.seq_lengths: Dict[int, int] = {}

        # Strides in element counts for computing flat GPU offsets.
        self._layer_stride = num_blocks * n_head * block_size * head_dim
        self._block_stride = n_head * block_size * head_dim
        self._head_stride  = block_size * head_dim

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def allocate_seq(self, seq_id: int) -> None:
        """Register a new sequence and allocate its first block."""
        if seq_id in self.block_tables:
            raise ValueError(f"seq_id {seq_id} is already active; call free_seq first")
        self.block_tables[seq_id] = [self._alloc_block()]
        self.seq_lengths[seq_id]  = 0

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

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        token_pos: int,
        k_vec: np.ndarray,
        v_vec: np.ndarray,
    ) -> None:
        """
        Write the K and V vectors for a single token position into the cache.

        Args:
            layer_idx:  Which transformer layer this K/V comes from.
            seq_id:     Sequence identifier.
            token_pos:  The absolute position of the token (0-indexed).
            k_vec:      numpy array of shape (n_head, head_dim).
            v_vec:      numpy array of shape (n_head, head_dim).
        """
        block_idx = token_pos // self.block_size
        slot_idx  = token_pos % self.block_size

        # Grow the block table on a block boundary.
        while block_idx >= len(self.block_tables[seq_id]):
            self.block_tables[seq_id].append(self._alloc_block())

        phys_block = self.block_tables[seq_id][block_idx]

        k_arr = np.asarray(k_vec, dtype=np.float32)
        v_arr = np.asarray(v_vec, dtype=np.float32)

        # Update CPU mirror.
        self.kv_k[layer_idx, phys_block, :, slot_idx, :] = k_arr
        self.kv_v[layer_idx, phys_block, :, slot_idx, :] = v_arr

        # Update GPU mirror incrementally (one contiguous head slice at a time).
        if _PYCUDA_AVAILABLE and self._kv_k_gpu is not None:
            base_k = (
                layer_idx  * self._layer_stride
                + phys_block * self._block_stride
                + slot_idx   * self.head_dim        # head=0, slot=slot_idx
            )
            base_v = base_k
            for h in range(self.n_head):
                off_k = (base_k + h * self._head_stride) * 4  # bytes
                off_v = (base_v + h * self._head_stride) * 4
                _cuda_driver.memcpy_htod(
                    int(self._kv_k_gpu) + off_k,
                    np.ascontiguousarray(k_arr[h]),
                )
                _cuda_driver.memcpy_htod(
                    int(self._kv_v_gpu) + off_v,
                    np.ascontiguousarray(v_arr[h]),
                )

    # ------------------------------------------------------------------
    # GPU device-pointer helpers (used by CudaKernelOps.paged_attention)
    # ------------------------------------------------------------------

    def device_ptr_k(self, layer_idx: int) -> ctypes.c_void_p:
        """Return a ctypes.c_void_p pointing to kv_k[layer_idx] on the GPU."""
        if not _PYCUDA_AVAILABLE or self._kv_k_gpu is None:
            raise RuntimeError(
                "pycuda is not available; cannot provide GPU device pointer"
            )
        offset = layer_idx * self._layer_stride * 4  # bytes
        return ctypes.c_void_p(int(self._kv_k_gpu) + offset)

    def device_ptr_v(self, layer_idx: int) -> ctypes.c_void_p:
        """Return a ctypes.c_void_p pointing to kv_v[layer_idx] on the GPU."""
        if not _PYCUDA_AVAILABLE or self._kv_v_gpu is None:
            raise RuntimeError(
                "pycuda is not available; cannot provide GPU device pointer"
            )
        offset = layer_idx * self._layer_stride * 4  # bytes
        return ctypes.c_void_p(int(self._kv_v_gpu) + offset)

    def has_device_mirror(self) -> bool:
        """Whether the GPU mirror exists and can be passed to the CUDA kernel."""
        return _PYCUDA_AVAILABLE and self._kv_k_gpu is not None and self._kv_v_gpu is not None

    # ------------------------------------------------------------------
    # Gather paths used when paged_attn.so is unavailable.
    # ------------------------------------------------------------------

    def gather_kv(
        self,
        layer_idx: int,
        seq_id: int,
        count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Copy the first `count` K/V vectors for a sequence into contiguous
        numpy arrays.  Used as a fallback when paged_attn.so is not available.

        Returns:
            k_out: (1, n_head, count, head_dim)
            v_out: (1, n_head, count, head_dim)
        """
        num_blocks_needed = (count + self.block_size - 1) // self.block_size
        phys_blocks = self.block_tables[seq_id][:num_blocks_needed]

        # Gather blocks from CPU mirror and flatten the token dimension.
        k_blocks = self.kv_k[layer_idx, phys_blocks]  # (nb, n_head, block_size, hd)
        v_blocks = self.kv_v[layer_idx, phys_blocks]

        # (nb, n_head, block_size, hd) -> (n_head, nb*block_size, hd)
        k_flat = k_blocks.transpose(1, 0, 2, 3).reshape(
            self.n_head, num_blocks_needed * self.block_size, self.head_dim
        )
        v_flat = v_blocks.transpose(1, 0, 2, 3).reshape(
            self.n_head, num_blocks_needed * self.block_size, self.head_dim
        )

        k_out = k_flat[:, :count, :][np.newaxis]  # (1, n_head, count, head_dim)
        v_out = v_flat[:, :count, :][np.newaxis]
        return k_out, v_out

    def gather_kv_padded(
        self,
        layer_idx: int,
        seq_ids: List[int],
        counts: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of gather_kv.  Zero-pads shorter sequences to max length.

        Returns:
            k_out: (B, n_head, max_count, head_dim), zero-padded
            v_out: (B, n_head, max_count, head_dim), zero-padded
        """
        B         = len(seq_ids)
        max_count = max(counts)

        k_out = np.zeros((B, self.n_head, max_count, self.head_dim), dtype=np.float32)
        v_out = np.zeros_like(k_out)

        for b, (seq_id, count) in enumerate(zip(seq_ids, counts)):
            k_b, v_b = self.gather_kv(layer_idx, seq_id, count)
            k_out[b, :, :count, :] = k_b[0]
            v_out[b, :, :count, :] = v_b[0]

        return k_out, v_out

    def get_block_tables_np(self, seq_ids: List[int]) -> np.ndarray:
        """
        Pack block tables into a contiguous int32 array for the C kernel.

        Returns:
            np.ndarray of shape (B, max_blocks_per_seq), dtype int32.
            Unused slots are filled with -1.
        """
        B          = len(seq_ids)
        max_blocks = max(len(self.block_tables[sid]) for sid in seq_ids)
        out        = np.full((B, max_blocks), -1, dtype=np.int32)
        for b, sid in enumerate(seq_ids):
            table = self.block_tables[sid]
            out[b, : len(table)] = table
        return out
