from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np

# Try to import pycuda; if unavailable, fall back to CPU-backed matrix multiply
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    PYCUDA_AVAILABLE = True
except Exception:
    cuda = None
    PYCUDA_AVAILABLE = False

# Load the shared library containing C ABI entrypoints declared in
# `src/combine.cu` under `extern "C"`.
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
datatype = np.float32
# Feature gate used by higher-level dispatch code.
HAS_FLASH_ATTENTION2_BWD = hasattr(lib, "FlashAttention2Backward")

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def flash_attention2_backward(
        dout: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        logsumexp: Tensor,
        causal: bool = False,
        softmax_scale: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """CUDA wrapper for FA2 backward.

        Notes
        -----
        - Requires `FlashAttention2Backward` symbol in `combine.so`.
        - Operates on contiguous host arrays and performs host<->device copies
          internally in the C++ wrapper (same style as existing kernels).
        """
        # Fast fail if caller requested CUDA path but library is stale.
        if not HAS_FLASH_ATTENTION2_BWD:
            raise RuntimeError(
                "FlashAttention2Backward CUDA symbol not found. "
                "Please rebuild kernels with compile_cuda.sh"
            )

        # Convert miniTorch tensors to contiguous host buffers.
        # Fast-path: when underlying storage is already contiguous host NumPy,
        # avoid extra materialization through `to_numpy()`.
        def _to_host_contiguous_np(x: Tensor) -> np.ndarray:
            td = x._tensor
            if td.is_contiguous() and isinstance(td._storage, np.ndarray):
                arr = td._storage
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                return np.ascontiguousarray(arr.reshape(x.shape), dtype=np.float32)
            return np.ascontiguousarray(x.to_numpy(), dtype=np.float32)

        dout_np = _to_host_contiguous_np(dout)
        q_np = _to_host_contiguous_np(q)
        k_np = _to_host_contiguous_np(k)
        v_np = _to_host_contiguous_np(v)
        out_np = _to_host_contiguous_np(out)
        lse_np = _to_host_contiguous_np(logsumexp)

        # Accept either (B,H,T) or (B,H,T,1) and canonicalize.
        if lse_np.ndim == 4 and lse_np.shape[-1] == 1:
            lse_np = lse_np[..., 0]

        if q_np.ndim != 4:
            raise ValueError(f"`q` must be rank-4 `(B,H,T,D)`, got {q_np.shape}")
        # Naming convention:
        # B = batch size, H = num heads, T = sequence length, D = head dim.
        bsz, nhead, seqlen, headdim = q_np.shape
        expected = (bsz, nhead, seqlen, headdim)
        for name, arr in (("k", k_np), ("v", v_np), ("out", out_np), ("dout", dout_np)):
            if arr.shape != expected:
                raise ValueError(f"`{name}` must have shape {expected}, got {arr.shape}")
        if lse_np.shape != (bsz, nhead, seqlen):
            raise ValueError(
                "`logsumexp` must have shape (B,H,T) or (B,H,T,1), "
                f"got {lse_np.shape}"
            )

        # Must match forward scale. Default aligns with standard attention.
        if softmax_scale is None:
            softmax_scale = 1.0 / np.sqrt(float(headdim))

        # Output gradients are fully written by CUDA wrapper.
        dq_np = np.empty_like(q_np, dtype=np.float32)
        dk_np = np.empty_like(k_np, dtype=np.float32)
        dv_np = np.empty_like(v_np, dtype=np.float32)

        # ABI contract for C function:
        # FlashAttention2Backward(float* dout, float* q, ..., int B,H,T,D,
        #                         int causal, float softmax_scale)
        lib.FlashAttention2Backward.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dout
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # q
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # k
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # v
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # lse
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dq
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dk
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # dv
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        lib.FlashAttention2Backward.restype = None

        # Flatten to 1D contiguous views; C side interprets logical shape via
        # explicit B/H/T/D arguments.
        lib.FlashAttention2Backward(
            dout_np.reshape(-1),
            q_np.reshape(-1),
            k_np.reshape(-1),
            v_np.reshape(-1),
            out_np.reshape(-1),
            lse_np.reshape(-1),
            dq_np.reshape(-1),
            dk_np.reshape(-1),
            dv_np.reshape(-1),
            bsz,
            nhead,
            seqlen,
            headdim,
            int(bool(causal)),
            float(softmax_scale),
        )

        # Keep backend consistent with caller expectations.
        backend = q.backend
        dq_t = tensor_from_numpy(np.ascontiguousarray(dq_np), backend=backend, requires_grad=False)
        dk_t = tensor_from_numpy(np.ascontiguousarray(dk_np), backend=backend, requires_grad=False)
        dv_t = tensor_from_numpy(np.ascontiguousarray(dv_np), backend=backend, requires_grad=False)
        return dq_t, dk_t, dv_t

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # If PyCUDA is unavailable, fall back to the CUDA-kernel-backed
        # `matrix_multiply` implementation which calls into the compiled
        # `MatrixMultiply` function (it handles host->device copies internally).
        if not PYCUDA_AVAILABLE:
            return CudaKernelOps.matrix_multiply(a, b)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out