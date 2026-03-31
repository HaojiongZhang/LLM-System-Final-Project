#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

#define MAX_DIMS 10
#define TILE 32
#define BASE_THREAD_NUM 32

#define ADD_FUNC       1
#define MUL_FUNC       2
#define ID_FUNC        3
#define NEG_FUNC       4
#define LT_FUNC        5
#define EQ_FUNC        6
#define SIGMOID_FUNC   7
#define RELU_FUNC      8
#define RELU_BACK_FUNC 9
#define LOG_FUNC       10
#define LOG_BACK_FUNC  11
#define EXP_FUNC       12
#define INV_FUNC       13
#define INV_BACK_FUNC  14
#define IS_CLOSE_FUNC  15
#define MAX_FUNC       16
#define POW            17
#define TANH           18

__device__ float fn(int fn_id, float x, float y=0) {
    switch(fn_id) {
      case ADD_FUNC: {
        return x + y;
      }
      case MUL_FUNC: {
        return x * y;
      }
      case ID_FUNC: {
      	return x;
      }
      case NEG_FUNC: {
        return -x;
      }
      case LT_FUNC: {
        if (x < y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case EQ_FUNC: {
        if (x == y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case SIGMOID_FUNC: {
        if (x >= 0) {
          return 1.0 / (1.0 + exp(-x));
        }
        else {
          return exp(x) / (1.0 + exp(x));
        }
      }
      case RELU_FUNC: {
        return max(x, 0.0);
      }
      case RELU_BACK_FUNC: {
        if (x > 0) {
          return y;
        }
        else {
          return 0.0;
        }
      }
      case LOG_FUNC: {
        return log(x + 1e-6);
      }
      case LOG_BACK_FUNC: {
        return y / (x + 1e-6);
      }
      case EXP_FUNC: {
        return exp(x);
      }
      case INV_FUNC: {
        return float(1.0 / x);
      }
      case INV_BACK_FUNC: {
        return -(1.0 / (x * x)) * y;
      }
      case IS_CLOSE_FUNC: {
        return (x - y < 1e-2) && (y - x < 1e-2);
      }
      case MAX_FUNC: {
        if (x > y) {
          return x;
        }
        else {
          return y;
        }
      }
      case POW: {
        return pow(x, y);
      }
      case TANH: {
        return tanh(x);
      }
      default: {
        return x + y;
      }
    }
    
}


__device__ int index_to_position(const int* index, const int* strides, int num_dims) {
    int position = 0;
    for (int i = 0; i < num_dims; ++i) {
        position += index[i] * strides[i];
    }
    return position;
}

__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims) {
    int cur_ord = ordinal;
    for (int i = num_dims - 1; i >= 0; --i) {
        int sh = shape[i];
        out_index[i] = cur_ord % sh;
        cur_ord /= sh;
    }
}

__device__ void broadcast_index(const int* big_index, const int* big_shape, const int* shape, int* out_index, int num_dims_big, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
        if (shape[i] > 1) {
            out_index[i] = big_index[i + (num_dims_big - num_dims)];
        } else {
            out_index[i] = 0;
        }
    }
}

__device__ __forceinline__ float WarpDotTensorCore16(
    const float* a,
    const float* b,
    int D,
    half* warp_a_tile,
    half* warp_b_tile,
    float* warp_c_tile) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  using namespace nvcuda::wmma;

  int lane = threadIdx.x & 31;
  float sum = 0.0f;

  for (int d0 = 0; d0 < D; d0 += 16) {
    // Build a tiny 16x16 x 16x16 MMA problem where only row 0 and col 0 are used,
    // so C[0,0] equals the 16-element dot product.
    for (int idx = lane; idx < 16 * 16; idx += 32) {
      warp_a_tile[idx] = __float2half(0.0f);
      warp_b_tile[idx] = __float2half(0.0f);
    }
    if (lane < 16) {
      warp_a_tile[lane] = __float2half(a[d0 + lane]);         // A(0, lane)
      warp_b_tile[lane * 16] = __float2half(b[d0 + lane]);    // B(lane, 0)
    }
    __syncwarp();

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    load_matrix_sync(a_frag, warp_a_tile, 16);
    load_matrix_sync(b_frag, warp_b_tile, 16);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(warp_c_tile, c_frag, 16, mem_row_major);

    if (lane == 0) {
      sum += warp_c_tile[0];
    }
    __syncwarp();
  }

  // Broadcast lane0 result to the full warp.
  sum = __shfl_sync(0xffffffff, sum, 0);
  return sum;
#else
  int lane = threadIdx.x & 31;
  float acc = 0.0f;
  for (int d = lane; d < D; d += 32) {
    acc += a[d] * b[d];
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  acc = __shfl_sync(0xffffffff, acc, 0);
  return acc;
#endif
}


__global__ void MatrixMultiplyKernel(
    float *out,
    const int *out_shape,
    const int *out_strides,
    float *a_storage,
    const int *a_shape,
    const int *a_strides,
    float *b_storage,
    const int *b_shape,
    const int *b_strides)
{
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch
   * format, with shape [batch_size, m, n], [batch_size, n, p].
   * Requirements:
   * - All data must be first moved to shared memory.
   * - Only read each cell in a and b once.
   * - Only write to global memory once per kernel.
   * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],
   * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]
   *
   * Args:
   *   out: compact 1D array of size batch_size x m x p to write the output to
   *   out_shape: shape of the output array
   *   out_strides: strides of the output array
   *   a_storage: compact 1D array of size batch_size x m x n
   *   a_shape: shape of the a array
   *   a_strides: strides of the a array
   *   b_storage: comapct 2D array of size batch_size x n x p
   *   b_shape: shape of the b array
   *   b_strides: strides of the b array
   *
   * Returns:
   *   None (Fills in out array)
   */

  __shared__ float a_shared[TILE][TILE];
  __shared__ float b_shared[TILE][TILE];

  // In each block, we will compute a batch of the output matrix
  // All the threads in the block will work together to compute this batch
  int batch = blockIdx.z;
  int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
  int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;

  /// BEGIN HW1_4
  /// TODO
  // Hints:
  // 1. Compute the row and column of the output matrix this block will compute
  // 2. Compute the position in the output array that this thread will write to
  // 3. Iterate over tiles of the two input matrices, read the data into shared memory
  // 4. Synchronize to make sure the data is available to all threads
  // 5. Compute the output tile for this thread block
  // 6. Synchronize to make sure all threads are done computing the output tile for (row, col)
  // 7. Write the output to global memory
  int m = a_shape[1];
  int n = a_shape[2];
  int p = b_shape[2]; 
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int out_batch_stride = out_shape[0] > 1 ? out_strides[0] : 0;

  
  // 移動指標到當前 batch 的起始位置
  float* a_ptr = a_storage + batch * a_batch_stride;
  float* b_ptr = b_storage + batch * b_batch_stride;
  float* out_ptr = out + batch * out_batch_stride;

  // 累加器 (Accumulator)
  float temp_sum = 0.0;

  for (int k = 0; k < n; k += TILE) {
      
      int col_a = k + ty;
      if (i < m && col_a < n) {
          int idx = i * a_strides[1] + col_a * a_strides[2];
          a_shared[tx][ty] = a_ptr[idx];
      } else {
          a_shared[tx][ty] = 0.0;
      }

      int row_b = k + tx;
      if (row_b < n && j < p) {
          int idx = row_b * b_strides[1] + j * b_strides[2];
          b_shared[tx][ty] = b_ptr[idx];
      } else {
          b_shared[tx][ty] = 0.0;
      }

      __syncthreads();

      for (int t = 0; t < TILE; ++t) {
          temp_sum += a_shared[tx][t] * b_shared[t][ty];
      }

      __syncthreads();
  }

  if (i < m && j < p) {
      int out_idx = i * out_strides[1] + j * out_strides[2];
      out_ptr[out_idx] = temp_sum;
  }
  /// END HW1_4
}


__global__ void mapKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    float *in_storage,
    int *in_shape,
    int *in_strides,
    int shape_size,
    int fn_id)
{
  /**
   * Map function. Apply a unary function to each element of the input array and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  in_storage: compact 1D array of size in_size
   *  in_shape: shape of the input array
   *  in_strides: strides of the input array
   *  shape_size: number of dimensions in the input and output arrays, assume dimensions are the same
   *  fn_id: id of the function to apply to each element of the input array
   *
   * Returns:
   *  None (Fills in out array)
   */

  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];

  /// BEGIN HW1_1
  /// TODO
  // Hints:
  // 1. Compute the position in the output array that this thread will write to
  // 2. Convert the position to the out_index according to out_shape
  // 3. Broadcast the out_index to the in_index according to in_shape (optional in some cases)
  // 4. Calculate the position of element in in_array according to in_index and in_strides
  // 5. Calculate the position of element in out_array according to out_index and out_strides
  // 6. Apply the unary function to the input element and write the output to the out memory

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < out_size) {
    int current_indices[MAX_DIMS];
    to_index(i, out_shape, out_index, shape_size);
    broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size);
    
    int in_pos = index_to_position(in_index, in_strides, shape_size);
    int out_pos = index_to_position(out_index, out_strides, shape_size);

    out[out_pos] = fn(fn_id, in_storage[in_pos]);
  }
  /// END HW1_1
}


__global__ void reduceKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    float *a_storage,
    int *a_shape,
    int *a_strides,
    int reduce_dim,
    float reduce_value,
    int shape_size,
    int fn_id)
{
  /**
   * Reduce function. Apply a reduce function to elements of the input array a and store the result in the output array.
   * Optimization:
   * Parallelize over the reduction operation. Each kernel performs one reduction.
   * e.g. a = [[1, 2, 3], [4, 5, 6]], kernel0 computes reduce([1, 2, 3]), kernel1 computes reduce([4, 5, 6]).
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  reduce_dim: dimension to reduce on
   *  reduce_value: initial value for the reduction
   *  shape_size: number of dimensions in the input & output array, assert dimensions are the same
   *  fn_id: id of the reduce function, currently only support add, multiply, and max
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

  // __shared__ double cache[BLOCK_DIM]; // Uncomment this line if you want to use shared memory to store partial results
  int out_index[MAX_DIMS];

  /// BEGIN HW1_3
  /// TODO
  // 1. Define the position of the output element that this thread or this block will write to
  // 2. Convert the out_pos to the out_index according to out_shape
  // 3. Initialize the reduce_value to the output element
  // 4. Iterate over the reduce_dim dimension of the input array to compute the reduced value
  // 5. Write the reduced value to out memory

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < out_size) {
    int out_index[MAX_DIMS];
    int in_index[MAX_DIMS];

    to_index(i, out_shape, out_index, shape_size);

    float curr = reduce_value;

    for (int k = 0; k < shape_size; k++) {
        in_index[k] = out_index[k];
    }
    int reduce_size = a_shape[reduce_dim];

    for (int k = 0; k < reduce_size; k++) {
        in_index[reduce_dim] = k;
        int in_pos = index_to_position(in_index, a_strides, shape_size);
        curr = fn(fn_id, curr, a_storage[in_pos]);
    }

    int out_pos = index_to_position(out_index, out_strides, shape_size);
    out[out_pos] = curr;
  }
  /// END HW1_3
}

__global__ void zipKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    int out_shape_size,
    float *a_storage,
    int *a_shape,
    int *a_strides,
    int a_shape_size,
    float *b_storage,
    int *b_shape,
    int *b_strides,
    int b_shape_size,
    int fn_id)
{
  /**
   * Zip function. Apply a binary function to elements of the input array a & b and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  out_shape_size: number of dimensions in the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  a_shape_size: number of dimensions in the input array
   *  b_storage: compact 1D array of size in_size
   *  b_shape: shape of the input array
   *  b_strides: strides of the input array
   *  b_shape_size: number of dimensions in the input array
   *  fn_id: id of the function to apply to each element of the a & b array
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

  int out_index[MAX_DIMS];
  int a_index[MAX_DIMS];
  int b_index[MAX_DIMS];

  /// BEGIN HW1_2
  /// TODO
  // Hints:
  // 1. Compute the position in the output array that this thread will write to
  // 2. Convert the position to the out_index according to out_shape
  // 3. Calculate the position of element in out_array according to out_index and out_strides
  // 4. Broadcast the out_index to the a_index according to a_shape
  // 5. Calculate the position of element in a_array according to a_index and a_strides
  // 6. Broadcast the out_index to the b_index according to b_shape
  // 7.Calculate the position of element in b_array according to b_index and b_strides
  // 8. Apply the binary function to the input elements in a_array & b_array and write the output to the out memory

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < out_size){
    to_index(i, out_shape, out_index, out_shape_size);
    broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size);
    broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size);
    int pos_out = index_to_position(out_index, out_strides, out_shape_size);
    int pos_a   = index_to_position(a_index, a_strides, a_shape_size);
    int pos_b   = index_to_position(b_index, b_strides, b_shape_size);

    out[pos_out] = fn(fn_id, a_storage[pos_a], b_storage[pos_b]);
  }
  /// END HW1_2
}


__global__ void FlashAttention2BackwardDQKernel(
  const float* __restrict__ dout,
  const float* __restrict__ q,
  const float* __restrict__ k,
  const float* __restrict__ v,
  const float* __restrict__ out,
  const float* __restrict__ lse,
  float* __restrict__ dq,
  int B,
  int H,
  int T,
  int D,
  int BQ,
  int BK,
  int use_tensor_core,
  int causal,
  float softmax_scale)
{
  int tile_id = blockIdx.x;
  int num_q_tiles = (T + BQ - 1) / BQ;
  int total_tiles = B * H * num_q_tiles;
  if (tile_id >= total_tiles) {
    return;
  }

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  int rem = tile_id;
  int q_tile_idx = rem % num_q_tiles;
  rem /= num_q_tiles;
  int h = rem % H;
  int b = rem / H;

  int q0 = q_tile_idx * BQ;
  int q_tile = min(BQ, T - q0);
  int D_PAD = D + 1;

  extern __shared__ float smem[];
  float* q_sh = smem;
  float* do_sh = q_sh + BQ * D_PAD;
  float* out_sh = do_sh + BQ * D_PAD;
  float* dq_acc = out_sh + BQ * D_PAD;
  float* k_sh = dq_acc + BQ * D_PAD;
  float* v_sh = k_sh + BK * D_PAD;
  float* Drow_sh = v_sh + BK * D_PAD;
  float* p_sh = Drow_sh + BQ;
  float* dS_sh = p_sh + BQ;
  float* lse_sh = dS_sh + BQ;

  int warps_per_block = blockDim.x / 32;
  half* tc_a_sh = reinterpret_cast<half*>(lse_sh + BQ);
  half* tc_b_sh = tc_a_sh + warps_per_block * 16 * 16;
  float* tc_c_sh = reinterpret_cast<float*>(tc_b_sh + warps_per_block * 16 * 16);

  if ((D & 3) == 0) {
    int D4 = D >> 2;
    for (int qi = 0; qi < q_tile; ++qi) {
      int row4 = ((b * H + h) * T + (q0 + qi)) * D;
      const float4* q4 = reinterpret_cast<const float4*>(q + row4);
      const float4* do4 = reinterpret_cast<const float4*>(dout + row4);
      const float4* out4 = reinterpret_cast<const float4*>(out + row4);
      for (int v4i = tid; v4i < D4; v4i += blockDim.x) {
        float4 qv = q4[v4i];
        float4 dov = do4[v4i];
        float4 ov = out4[v4i];
        int d = v4i << 2;
        q_sh[qi * D_PAD + d + 0] = qv.x;
        q_sh[qi * D_PAD + d + 1] = qv.y;
        q_sh[qi * D_PAD + d + 2] = qv.z;
        q_sh[qi * D_PAD + d + 3] = qv.w;
        do_sh[qi * D_PAD + d + 0] = dov.x;
        do_sh[qi * D_PAD + d + 1] = dov.y;
        do_sh[qi * D_PAD + d + 2] = dov.z;
        do_sh[qi * D_PAD + d + 3] = dov.w;
        out_sh[qi * D_PAD + d + 0] = ov.x;
        out_sh[qi * D_PAD + d + 1] = ov.y;
        out_sh[qi * D_PAD + d + 2] = ov.z;
        out_sh[qi * D_PAD + d + 3] = ov.w;
      }
    }
  } else {
    for (int linear = tid; linear < q_tile * D; linear += blockDim.x) {
      int qi = linear / D;
      int d = linear % D;
      int row4 = ((b * H + h) * T + (q0 + qi)) * D;
      q_sh[qi * D_PAD + d] = q[row4 + d];
      do_sh[qi * D_PAD + d] = dout[row4 + d];
      out_sh[qi * D_PAD + d] = out[row4 + d];
    }
  }

  for (int linear = tid; linear < q_tile * D_PAD; linear += blockDim.x) {
    int d = linear % D_PAD;
    if (d < D) {
      dq_acc[linear] = 0.0f;
    }
  }
  __syncthreads();

  if (warp_id < q_tile) {
    int i = q0 + warp_id;
    if (lane == 0) {
      lse_sh[warp_id] = lse[(b * H + h) * T + i];
    }
    float part_D = 0.0f;
    for (int d = lane; d < D; d += 32) {
      part_D += do_sh[warp_id * D_PAD + d] * out_sh[warp_id * D_PAD + d];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      part_D += __shfl_down_sync(0xffffffff, part_D, offset);
    }
    if (lane == 0) {
      Drow_sh[warp_id] = part_D;
    }
  }
  __syncthreads();

  for (int k0 = 0; k0 < T; k0 += BK) {
    if (causal) {
      int q_max = q0 + q_tile - 1;
      if (k0 > q_max) {
        break;
      }
    }
    int k_tile = min(BK, T - k0);

    if ((D & 3) == 0) {
      int D4 = D >> 2;
      for (int tj = 0; tj < k_tile; ++tj) {
        int col4 = ((b * H + h) * T + (k0 + tj)) * D;
        const float4* k4 = reinterpret_cast<const float4*>(k + col4);
        const float4* v4 = reinterpret_cast<const float4*>(v + col4);
        for (int v4i = tid; v4i < D4; v4i += blockDim.x) {
          float4 kv = k4[v4i];
          float4 vv = v4[v4i];
          int d = v4i << 2;
          k_sh[tj * D_PAD + d + 0] = kv.x;
          k_sh[tj * D_PAD + d + 1] = kv.y;
          k_sh[tj * D_PAD + d + 2] = kv.z;
          k_sh[tj * D_PAD + d + 3] = kv.w;
          v_sh[tj * D_PAD + d + 0] = vv.x;
          v_sh[tj * D_PAD + d + 1] = vv.y;
          v_sh[tj * D_PAD + d + 2] = vv.z;
          v_sh[tj * D_PAD + d + 3] = vv.w;
        }
      }
    } else {
      for (int linear = tid; linear < k_tile * D; linear += blockDim.x) {
        int tj = linear / D;
        int d = linear % D;
        int col4 = ((b * H + h) * T + (k0 + tj)) * D;
        k_sh[tj * D_PAD + d] = k[col4 + d];
        v_sh[tj * D_PAD + d] = v[col4 + d];
      }
    }
    __syncthreads();

    for (int tj = 0; tj < k_tile; ++tj) {
      if (warp_id < q_tile) {
        int j = k0 + tj;
        int i = q0 + warp_id;
        if (!causal || j <= i) {
          float part_score = 0.0f;
          float part_dP = 0.0f;
          const float* q_row = &q_sh[warp_id * D_PAD];
          const float* do_row = &do_sh[warp_id * D_PAD];
          const float* k_row = &k_sh[tj * D_PAD];
          const float* v_row = &v_sh[tj * D_PAD];
          bool use_tc = (use_tensor_core != 0) && ((D & 15) == 0);
          if (use_tc) {
            half* warp_a_tile = tc_a_sh + warp_id * 16 * 16;
            half* warp_b_tile = tc_b_sh + warp_id * 16 * 16;
            float* warp_c_tile = tc_c_sh + warp_id * 16 * 16;
            part_score = WarpDotTensorCore16(q_row, k_row, D, warp_a_tile, warp_b_tile, warp_c_tile);
            part_dP = WarpDotTensorCore16(do_row, v_row, D, warp_a_tile, warp_b_tile, warp_c_tile);
          } else {
            for (int d = lane; d < D; d += 32) {
              part_score += q_row[d] * k_row[d];
              part_dP += do_row[d] * v_row[d];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
              part_score += __shfl_down_sync(0xffffffff, part_score, offset);
              part_dP += __shfl_down_sync(0xffffffff, part_dP, offset);
            }
          }
          if (lane == 0) {
            float p = expf(part_score * softmax_scale - lse_sh[warp_id]);
            p_sh[warp_id] = p;
            dS_sh[warp_id] = p * (part_dP - Drow_sh[warp_id]);
          }
        } else if (lane == 0) {
          p_sh[warp_id] = 0.0f;
          dS_sh[warp_id] = 0.0f;
        }
      }
      __syncthreads();

      if (warp_id < q_tile) {
        float dS = dS_sh[warp_id];
        for (int d = lane; d < D; d += 32) {
          dq_acc[warp_id * D_PAD + d] += dS * k_sh[tj * D_PAD + d] * softmax_scale;
        }
      }
      __syncthreads();
    }
  }

  if ((D & 3) == 0) {
    int D4 = D >> 2;
    for (int qi = 0; qi < q_tile; ++qi) {
      int row4 = ((b * H + h) * T + (q0 + qi)) * D;
      float4* dq4 = reinterpret_cast<float4*>(dq + row4);
      for (int v4i = tid; v4i < D4; v4i += blockDim.x) {
        int d = v4i << 2;
        float4 outv;
        outv.x = dq_acc[qi * D_PAD + d + 0];
        outv.y = dq_acc[qi * D_PAD + d + 1];
        outv.z = dq_acc[qi * D_PAD + d + 2];
        outv.w = dq_acc[qi * D_PAD + d + 3];
        dq4[v4i] = outv;
      }
    }
  } else {
    for (int linear = tid; linear < q_tile * D; linear += blockDim.x) {
      int qi = linear / D;
      int d = linear % D;
      dq[((b * H + h) * T + (q0 + qi)) * D + d] = dq_acc[qi * D_PAD + d];
    }
  }
}

__global__ void FlashAttention2BackwardDKDVKernel(
  const float* __restrict__ dout,
  const float* __restrict__ q,
  const float* __restrict__ k,
  const float* __restrict__ v,
  const float* __restrict__ out,
  const float* __restrict__ lse,
  float* __restrict__ dk,
  float* __restrict__ dv,
  int B,
  int H,
  int T,
  int D,
  int BQ,
  int BK,
  int use_tensor_core,
  int causal,
  float softmax_scale)
{
  int tile_id = blockIdx.x;
  int num_k_tiles = (T + BK - 1) / BK;
  int total_tiles = B * H * num_k_tiles;
  if (tile_id >= total_tiles) {
    return;
  }

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int warps_per_block = blockDim.x / 32;

  int rem = tile_id;
  int k_tile_idx = rem % num_k_tiles;
  rem /= num_k_tiles;
  int h = rem % H;
  int b = rem / H;

  int k0 = k_tile_idx * BK;
  int k_tile = min(BK, T - k0);
  int D_PAD = D + 1;

  extern __shared__ float smem[];
  float* k_sh = smem;
  float* v_sh = k_sh + BK * D_PAD;
  float* dk_acc = v_sh + BK * D_PAD;
  float* dv_acc = dk_acc + BK * D_PAD;
  float* q_sh = dv_acc + BK * D_PAD;
  float* do_sh = q_sh + BQ * D_PAD;
  float* out_sh = do_sh + BQ * D_PAD;
  float* Drow_sh = out_sh + BQ * D_PAD;
  float* lse_sh = Drow_sh + BQ;
  float* p_sh = lse_sh + BQ;
  float* dS_sh = p_sh + warps_per_block * BQ;

  half* tc_a_sh = reinterpret_cast<half*>(dS_sh + warps_per_block * BQ);
  half* tc_b_sh = tc_a_sh + warps_per_block * 16 * 16;
  float* tc_c_sh = reinterpret_cast<float*>(tc_b_sh + warps_per_block * 16 * 16);

  if ((D & 3) == 0) {
    int D4 = D >> 2;
    for (int tj = 0; tj < k_tile; ++tj) {
      int row4 = ((b * H + h) * T + (k0 + tj)) * D;
      const float4* k4 = reinterpret_cast<const float4*>(k + row4);
      const float4* v4 = reinterpret_cast<const float4*>(v + row4);
      for (int v4i = tid; v4i < D4; v4i += blockDim.x) {
        float4 kv = k4[v4i];
        float4 vv = v4[v4i];
        int d = v4i << 2;
        k_sh[tj * D_PAD + d + 0] = kv.x;
        k_sh[tj * D_PAD + d + 1] = kv.y;
        k_sh[tj * D_PAD + d + 2] = kv.z;
        k_sh[tj * D_PAD + d + 3] = kv.w;
        v_sh[tj * D_PAD + d + 0] = vv.x;
        v_sh[tj * D_PAD + d + 1] = vv.y;
        v_sh[tj * D_PAD + d + 2] = vv.z;
        v_sh[tj * D_PAD + d + 3] = vv.w;
      }
    }
  } else {
    for (int linear = tid; linear < k_tile * D; linear += blockDim.x) {
      int tj = linear / D;
      int d = linear % D;
      int row4 = ((b * H + h) * T + (k0 + tj)) * D;
      k_sh[tj * D_PAD + d] = k[row4 + d];
      v_sh[tj * D_PAD + d] = v[row4 + d];
    }
  }

  for (int linear = tid; linear < k_tile * D_PAD; linear += blockDim.x) {
    int d = linear % D_PAD;
    if (d < D) {
      dk_acc[linear] = 0.0f;
      dv_acc[linear] = 0.0f;
    }
  }
  __syncthreads();

  for (int q0 = 0; q0 < T; q0 += BQ) {
    int q_tile = min(BQ, T - q0);
    if (causal && (q0 + q_tile - 1) < k0) {
      continue;
    }

    if ((D & 3) == 0) {
      int D4 = D >> 2;
      for (int qi = 0; qi < q_tile; ++qi) {
        int row4 = ((b * H + h) * T + (q0 + qi)) * D;
        const float4* q4 = reinterpret_cast<const float4*>(q + row4);
        const float4* do4 = reinterpret_cast<const float4*>(dout + row4);
        const float4* out4 = reinterpret_cast<const float4*>(out + row4);
        for (int v4i = tid; v4i < D4; v4i += blockDim.x) {
          float4 qv = q4[v4i];
          float4 dov = do4[v4i];
          float4 ov = out4[v4i];
          int d = v4i << 2;
          q_sh[qi * D_PAD + d + 0] = qv.x;
          q_sh[qi * D_PAD + d + 1] = qv.y;
          q_sh[qi * D_PAD + d + 2] = qv.z;
          q_sh[qi * D_PAD + d + 3] = qv.w;
          do_sh[qi * D_PAD + d + 0] = dov.x;
          do_sh[qi * D_PAD + d + 1] = dov.y;
          do_sh[qi * D_PAD + d + 2] = dov.z;
          do_sh[qi * D_PAD + d + 3] = dov.w;
          out_sh[qi * D_PAD + d + 0] = ov.x;
          out_sh[qi * D_PAD + d + 1] = ov.y;
          out_sh[qi * D_PAD + d + 2] = ov.z;
          out_sh[qi * D_PAD + d + 3] = ov.w;
        }
      }
    } else {
      for (int linear = tid; linear < q_tile * D; linear += blockDim.x) {
        int qi = linear / D;
        int d = linear % D;
        int row4 = ((b * H + h) * T + (q0 + qi)) * D;
        q_sh[qi * D_PAD + d] = q[row4 + d];
        do_sh[qi * D_PAD + d] = dout[row4 + d];
        out_sh[qi * D_PAD + d] = out[row4 + d];
      }
    }
    __syncthreads();

    if (warp_id < q_tile) {
      int i = q0 + warp_id;
      if (lane == 0) {
        lse_sh[warp_id] = lse[(b * H + h) * T + i];
      }
      float part_D = 0.0f;
      for (int d = lane; d < D; d += 32) {
        part_D += do_sh[warp_id * D_PAD + d] * out_sh[warp_id * D_PAD + d];
      }
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        part_D += __shfl_down_sync(0xffffffff, part_D, offset);
      }
      if (lane == 0) {
        Drow_sh[warp_id] = part_D;
      }
    }
    __syncthreads();

    for (int tj = warp_id; tj < k_tile; tj += warps_per_block) {
      int j = k0 + tj;
      bool use_tc = (use_tensor_core != 0) && ((D & 15) == 0);
      float* warp_p = p_sh + warp_id * BQ;
      float* warp_dS = dS_sh + warp_id * BQ;
      for (int qi = 0; qi < q_tile; ++qi) {
        int i = q0 + qi;
        if (causal && j > i) {
          if (lane == 0) {
            warp_p[qi] = 0.0f;
            warp_dS[qi] = 0.0f;
          }
          continue;
        }
        float score;
        float dP;
        const float* q_row = &q_sh[qi * D_PAD];
        const float* do_row = &do_sh[qi * D_PAD];
        const float* k_row = &k_sh[tj * D_PAD];
        const float* v_row = &v_sh[tj * D_PAD];
        if (use_tc) {
          half* warp_a_tile = tc_a_sh + warp_id * 16 * 16;
          half* warp_b_tile = tc_b_sh + warp_id * 16 * 16;
          float* warp_c_tile = tc_c_sh + warp_id * 16 * 16;
          score = WarpDotTensorCore16(q_row, k_row, D, warp_a_tile, warp_b_tile, warp_c_tile);
          dP = WarpDotTensorCore16(do_row, v_row, D, warp_a_tile, warp_b_tile, warp_c_tile);
        } else {
          float part_score = 0.0f;
          float part_dP = 0.0f;
          for (int dd = lane; dd < D; dd += 32) {
            part_score += q_row[dd] * k_row[dd];
            part_dP += do_row[dd] * v_row[dd];
          }
          #pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            part_score += __shfl_down_sync(0xffffffff, part_score, offset);
            part_dP += __shfl_down_sync(0xffffffff, part_dP, offset);
          }
          score = __shfl_sync(0xffffffff, part_score, 0);
          dP = __shfl_sync(0xffffffff, part_dP, 0);
        }
        if (lane == 0) {
          float p = expf(score * softmax_scale - lse_sh[qi]);
          warp_p[qi] = p;
          warp_dS[qi] = p * (dP - Drow_sh[qi]);
        }
      }
      __syncwarp();

      for (int d = lane; d < D; d += 32) {
        float dk_sum = 0.0f;
        float dv_sum = 0.0f;
        for (int qi = 0; qi < q_tile; ++qi) {
          float p = warp_p[qi];
          float dS = warp_dS[qi];
          dk_sum += dS * q_sh[qi * D_PAD + d] * softmax_scale;
          dv_sum += p * do_sh[qi * D_PAD + d];
        }
        dk_acc[tj * D_PAD + d] += dk_sum;
        dv_acc[tj * D_PAD + d] += dv_sum;
      }
      __syncwarp();
    }
    __syncthreads();
  }

  for (int linear = tid; linear < k_tile * D; linear += blockDim.x) {
    int tj = linear / D;
    int d = linear % D;
    int row4 = ((b * H + h) * T + (k0 + tj)) * D;
    dk[row4 + d] = dk_acc[tj * D_PAD + d];
    dv[row4 + d] = dv_acc[tj * D_PAD + d];
  }
}

__global__ void DenseAttentionDRowKernel(
  const float* __restrict__ dout,
  const float* __restrict__ out,
  float* __restrict__ drow,
  int rows,
  int D) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  int base = row * D;
  float acc = 0.0f;
  for (int d = 0; d < D; ++d) {
    acc += dout[base + d] * out[base + d];
  }
  drow[row] = acc;
}

__global__ void DenseAttentionMaterializeKernel(
  const float* __restrict__ dout,
  const float* __restrict__ q,
  const float* __restrict__ k,
  const float* __restrict__ v,
  const float* __restrict__ lse,
  const float* __restrict__ drow,
  float* __restrict__ probs,
  float* __restrict__ dP,
  float* __restrict__ dS,
  int rows,
  int T,
  int D,
  int causal,
  float softmax_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * T;
  if (idx >= total) {
    return;
  }

  int row = idx / T;
  int j = idx % T;
  int bh = row / T;
  int i = row % T;
  int matrix_offset = bh * T * T + i * T + j;

  if (causal && j > i) {
    probs[matrix_offset] = 0.0f;
    dP[matrix_offset] = 0.0f;
    dS[matrix_offset] = 0.0f;
    return;
  }

  int q_base = row * D;
  int kv_base = (bh * T + j) * D;
  float score = 0.0f;
  float dp_val = 0.0f;
  for (int d = 0; d < D; ++d) {
    score += q[q_base + d] * k[kv_base + d];
    dp_val += dout[q_base + d] * v[kv_base + d];
  }

  float p = expf(score * softmax_scale - lse[row]);
  probs[matrix_offset] = p;
  dP[matrix_offset] = dp_val;
  dS[matrix_offset] = p * (dp_val - drow[row]);
}

__global__ void DenseAttentionDVKernel(
  const float* __restrict__ probs,
  const float* __restrict__ dout,
  float* __restrict__ dv,
  int rows,
  int T,
  int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D;
  if (idx >= total) {
    return;
  }

  int row = idx / D;
  int d = idx % D;
  int bh = row / T;
  int j = row % T;
  int matrix_base = bh * T * T;
  int dout_base = bh * T * D + d;
  float acc = 0.0f;
  for (int i = 0; i < T; ++i) {
    acc += probs[matrix_base + i * T + j] * dout[dout_base + i * D];
  }
  dv[row * D + d] = acc;
}

__global__ void DenseAttentionDQKernel(
  const float* __restrict__ dS,
  const float* __restrict__ k,
  float* __restrict__ dq,
  int rows,
  int T,
  int D,
  float softmax_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D;
  if (idx >= total) {
    return;
  }

  int row = idx / D;
  int d = idx % D;
  int bh = row / T;
  int i = row % T;
  int matrix_base = bh * T * T + i * T;
  int k_base = bh * T * D + d;
  float acc = 0.0f;
  for (int j = 0; j < T; ++j) {
    acc += dS[matrix_base + j] * k[k_base + j * D];
  }
  dq[row * D + d] = acc * softmax_scale;
}

__global__ void DenseAttentionDKKernel(
  const float* __restrict__ dS,
  const float* __restrict__ q,
  float* __restrict__ dk,
  int rows,
  int T,
  int D,
  float softmax_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * D;
  if (idx >= total) {
    return;
  }

  int row = idx / D;
  int d = idx % D;
  int bh = row / T;
  int j = row % T;
  int matrix_base = bh * T * T + j;
  int q_base = bh * T * D + d;
  float acc = 0.0f;
  for (int i = 0; i < T; ++i) {
    acc += dS[matrix_base + i * T] * q[q_base + i * D];
  }
  dk[row * D + d] = acc * softmax_scale;
}


extern "C" {

// Persistent CUDA buffers for FA2 backward wrapper.
// This avoids cudaMalloc/cudaFree on every Python call, which is expensive
// and can dominate end-to-end timing for moderate sequence lengths.
static float* g_fa2_dout = nullptr;
static float* g_fa2_q = nullptr;
static float* g_fa2_k = nullptr;
static float* g_fa2_v = nullptr;
static float* g_fa2_out = nullptr;
static float* g_fa2_lse = nullptr;
static float* g_fa2_dq = nullptr;
static float* g_fa2_dk = nullptr;
static float* g_fa2_dv = nullptr;
static size_t g_fa2_cap4 = 0;
static size_t g_fa2_cap3 = 0;
static int g_fa2_device = -1;

static float* g_dense_dout = nullptr;
static float* g_dense_q = nullptr;
static float* g_dense_k = nullptr;
static float* g_dense_v = nullptr;
static float* g_dense_out = nullptr;
static float* g_dense_lse = nullptr;
static float* g_dense_dq = nullptr;
static float* g_dense_dk = nullptr;
static float* g_dense_dv = nullptr;
static float* g_dense_drow = nullptr;
static float* g_dense_probs = nullptr;
static float* g_dense_dp = nullptr;
static float* g_dense_ds = nullptr;
static size_t g_dense_cap4 = 0;
static size_t g_dense_cap3 = 0;
static size_t g_dense_cap2 = 0;
static int g_dense_device = -1;

static void FreeFA2Buffers() {
  if (g_fa2_dout) cudaFree(g_fa2_dout);
  if (g_fa2_q) cudaFree(g_fa2_q);
  if (g_fa2_k) cudaFree(g_fa2_k);
  if (g_fa2_v) cudaFree(g_fa2_v);
  if (g_fa2_out) cudaFree(g_fa2_out);
  if (g_fa2_lse) cudaFree(g_fa2_lse);
  if (g_fa2_dq) cudaFree(g_fa2_dq);
  if (g_fa2_dk) cudaFree(g_fa2_dk);
  if (g_fa2_dv) cudaFree(g_fa2_dv);

  g_fa2_dout = nullptr;
  g_fa2_q = nullptr;
  g_fa2_k = nullptr;
  g_fa2_v = nullptr;
  g_fa2_out = nullptr;
  g_fa2_lse = nullptr;
  g_fa2_dq = nullptr;
  g_fa2_dk = nullptr;
  g_fa2_dv = nullptr;
  g_fa2_cap4 = 0;
  g_fa2_cap3 = 0;
  g_fa2_device = -1;
}

static void FreeDenseBuffers() {
  if (g_dense_dout) cudaFree(g_dense_dout);
  if (g_dense_q) cudaFree(g_dense_q);
  if (g_dense_k) cudaFree(g_dense_k);
  if (g_dense_v) cudaFree(g_dense_v);
  if (g_dense_out) cudaFree(g_dense_out);
  if (g_dense_lse) cudaFree(g_dense_lse);
  if (g_dense_dq) cudaFree(g_dense_dq);
  if (g_dense_dk) cudaFree(g_dense_dk);
  if (g_dense_dv) cudaFree(g_dense_dv);
  if (g_dense_drow) cudaFree(g_dense_drow);
  if (g_dense_probs) cudaFree(g_dense_probs);
  if (g_dense_dp) cudaFree(g_dense_dp);
  if (g_dense_ds) cudaFree(g_dense_ds);

  g_dense_dout = nullptr;
  g_dense_q = nullptr;
  g_dense_k = nullptr;
  g_dense_v = nullptr;
  g_dense_out = nullptr;
  g_dense_lse = nullptr;
  g_dense_dq = nullptr;
  g_dense_dk = nullptr;
  g_dense_dv = nullptr;
  g_dense_drow = nullptr;
  g_dense_probs = nullptr;
  g_dense_dp = nullptr;
  g_dense_ds = nullptr;
  g_dense_cap4 = 0;
  g_dense_cap3 = 0;
  g_dense_cap2 = 0;
  g_dense_device = -1;
}

void FlashAttention2Backward(
    float* dout,
    float* q,
    float* k,
    float* v,
    float* out,
    float* lse,
    float* dq,
    float* dk,
    float* dv,
    int B,
    int H,
    int T,
    int D,
    int causal,
    float softmax_scale) {
    // Host-side C ABI wrapper called from Python ctypes.
    // Performs:
    //   1) H2D copies,
    //   2) kernel launch,
    //   3) D2H copies,
    //   4) cleanup.
    size_t size4 = (size_t)B * H * T * D;
    size_t size3 = (size_t)B * H * T;

    // Ensure buffers belong to active device; reallocate if device changed.
    int device = 0;
    cudaGetDevice(&device);
    if (g_fa2_device != -1 && g_fa2_device != device) {
      FreeFA2Buffers();
    }

    // Grow-on-demand allocation strategy.
    if (size4 > g_fa2_cap4 || size3 > g_fa2_cap3 || g_fa2_dout == nullptr) {
      FreeFA2Buffers();

      cudaMalloc(&g_fa2_dout, size4 * sizeof(float));
      cudaMalloc(&g_fa2_q, size4 * sizeof(float));
      cudaMalloc(&g_fa2_k, size4 * sizeof(float));
      cudaMalloc(&g_fa2_v, size4 * sizeof(float));
      cudaMalloc(&g_fa2_out, size4 * sizeof(float));
      cudaMalloc(&g_fa2_lse, size3 * sizeof(float));
      cudaMalloc(&g_fa2_dq, size4 * sizeof(float));
      cudaMalloc(&g_fa2_dk, size4 * sizeof(float));
      cudaMalloc(&g_fa2_dv, size4 * sizeof(float));

      g_fa2_cap4 = size4;
      g_fa2_cap3 = size3;
      g_fa2_device = device;
    }

    float *d_dout = g_fa2_dout, *d_q = g_fa2_q, *d_k = g_fa2_k, *d_v = g_fa2_v;
    float *d_out = g_fa2_out, *d_lse = g_fa2_lse;
    float *d_dq = g_fa2_dq, *d_dk = g_fa2_dk, *d_dv = g_fa2_dv;

    cudaStream_t stream = 0;

    // Copy host tensors to device.
    cudaMemcpyAsync(d_dout, dout, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_q, q, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_k, k, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v, v, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out, out, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lse, lse, size3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Architecture-aware 2D tiling parameters (Q tile x K tile).
    // - sm80+ (Ampere/Hopper): prefer larger BK for better reuse.
    // - sm70/75 (Volta/Turing): keep BK moderate to balance occupancy/shared memory.
    // - legacy: conservative defaults.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int BQ = (D <= 64) ? 8 : 4;   // warps per block (one warp per query row)
    int BK = (D <= 64) ? 32 : 16; // default K/V tile size

    if (prop.major >= 8) {
      BK = (D <= 64) ? 64 : 32;
    } else if (prop.major == 7) {
      BK = (D <= 64) ? 32 : 16;
    } else {
      BK = (D <= 64) ? 16 : 8;
      BQ = (D <= 64) ? 4 : 2;
    }

    int threadsDQ = BQ * 32;
    if (threadsDQ > 256) {
      threadsDQ = 256;
    }
    int threadsDKDV = 256;

    int num_q_tiles = (T + BQ - 1) / BQ;
    int num_k_tiles = (T + BK - 1) / BK;
    int blocksDQ = B * H * num_q_tiles;
    int blocksDKDV = B * H * num_k_tiles;

    int D_PAD = D + 1;
    size_t sharedFloatsDQ = (size_t)(4 * BQ * D_PAD + 2 * BK * D_PAD + 4 * BQ);
    size_t baseSharedBytesDQ = sharedFloatsDQ * sizeof(float);
    size_t sharedFloatsDKDV = (size_t)(4 * BK * D_PAD + 3 * BQ * D_PAD + 2 * BQ + 2 * (threadsDKDV / 32) * BQ);
    size_t baseSharedBytesDKDV = sharedFloatsDKDV * sizeof(float);

    // Tensor-core scratch is optional. On V100 (sm70), this WMMA micro-path
    // may reduce occupancy due to extra shared memory and can be slower than
    // scalar/shuffle dots in this kernel structure.
    int use_tensor_core = 0;
    if (prop.major >= 7 && (D % 16 == 0) && D >= 128) {
      use_tensor_core = 1;
    }

    size_t tcSharedBytesDQ = 0;
    size_t tcSharedBytesDKDV = 0;
    if (use_tensor_core) {
      int dqWarps = threadsDQ / 32;
      int dkdvWarps = threadsDKDV / 32;
      tcSharedBytesDQ =
        (size_t)dqWarps * (16 * 16 * sizeof(half)) +
        (size_t)dqWarps * (16 * 16 * sizeof(half)) +
        (size_t)dqWarps * (16 * 16 * sizeof(float));
      tcSharedBytesDKDV =
        (size_t)dkdvWarps * (16 * 16 * sizeof(half)) +
        (size_t)dkdvWarps * (16 * 16 * sizeof(half)) +
        (size_t)dkdvWarps * (16 * 16 * sizeof(float));
    }
    size_t sharedBytesDQ = baseSharedBytesDQ + tcSharedBytesDQ;
    size_t sharedBytesDKDV = baseSharedBytesDKDV + tcSharedBytesDKDV;

    cudaFuncSetAttribute(
        FlashAttention2BackwardDQKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)sharedBytesDQ);
    cudaFuncSetAttribute(
        FlashAttention2BackwardDKDVKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)sharedBytesDKDV);

    FlashAttention2BackwardDQKernel<<<blocksDQ, threadsDQ, sharedBytesDQ, stream>>>(
        d_dout,
        d_q,
        d_k,
        d_v,
        d_out,
        d_lse,
        d_dq,
        B,
        H,
        T,
        D,
        BQ,
        BK,
        use_tensor_core,
        causal,
        softmax_scale);
      FlashAttention2BackwardDKDVKernel<<<blocksDKDV, threadsDKDV, sharedBytesDKDV, stream>>>(
        d_dout,
        d_q,
        d_k,
        d_v,
        d_out,
        d_lse,
        d_dk,
        d_dv,
        B,
        H,
        T,
        D,
        BQ,
        BK,
        use_tensor_core,
        causal,
        softmax_scale);

    // Surface launch errors early.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "FlashAttention2Backward Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Copy gradients back to host buffers expected by ctypes caller.
    cudaMemcpyAsync(dq, d_dq, size4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(dk, d_dk, size4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(dv, d_dv, size4 * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // One sync for kernel + all copies.
    cudaStreamSynchronize(stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "FlashAttention2Backward Runtime Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}

void DenseAttentionBackward(
    float* dout,
    float* q,
    float* k,
    float* v,
    float* out,
    float* lse,
    float* dq,
    float* dk,
    float* dv,
    int B,
    int H,
    int T,
    int D,
    int causal,
    float softmax_scale) {
    size_t size4 = (size_t)B * H * T * D;
    size_t size3 = (size_t)B * H * T;
    size_t size2 = (size_t)B * H * T * T;

    int device = 0;
    cudaGetDevice(&device);
    if (g_dense_device != -1 && g_dense_device != device) {
      FreeDenseBuffers();
    }

    if (size4 > g_dense_cap4 || size3 > g_dense_cap3 || size2 > g_dense_cap2 || g_dense_dout == nullptr) {
      FreeDenseBuffers();

      cudaMalloc(&g_dense_dout, size4 * sizeof(float));
      cudaMalloc(&g_dense_q, size4 * sizeof(float));
      cudaMalloc(&g_dense_k, size4 * sizeof(float));
      cudaMalloc(&g_dense_v, size4 * sizeof(float));
      cudaMalloc(&g_dense_out, size4 * sizeof(float));
      cudaMalloc(&g_dense_lse, size3 * sizeof(float));
      cudaMalloc(&g_dense_dq, size4 * sizeof(float));
      cudaMalloc(&g_dense_dk, size4 * sizeof(float));
      cudaMalloc(&g_dense_dv, size4 * sizeof(float));
      cudaMalloc(&g_dense_drow, size3 * sizeof(float));
      cudaMalloc(&g_dense_probs, size2 * sizeof(float));
      cudaMalloc(&g_dense_dp, size2 * sizeof(float));
      cudaMalloc(&g_dense_ds, size2 * sizeof(float));

      g_dense_cap4 = size4;
      g_dense_cap3 = size3;
      g_dense_cap2 = size2;
      g_dense_device = device;
    }

    float *d_dout = g_dense_dout, *d_q = g_dense_q, *d_k = g_dense_k, *d_v = g_dense_v;
    float *d_out = g_dense_out, *d_lse = g_dense_lse;
    float *d_dq = g_dense_dq, *d_dk = g_dense_dk, *d_dv = g_dense_dv;
    float *d_drow = g_dense_drow, *d_probs = g_dense_probs, *d_dp = g_dense_dp, *d_ds = g_dense_ds;

    cudaStream_t stream = 0;
    cudaMemcpyAsync(d_dout, dout, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_q, q, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_k, k, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v, v, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out, out, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lse, lse, size3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    int rows = B * H * T;
    int threads = 256;
    int blocks_rows = (rows + threads - 1) / threads;
    int blocks_matrix = ((int)(rows * T) + threads - 1) / threads;
    int blocks_vec = ((int)(rows * D) + threads - 1) / threads;

    DenseAttentionDRowKernel<<<blocks_rows, threads, 0, stream>>>(d_dout, d_out, d_drow, rows, D);
    DenseAttentionMaterializeKernel<<<blocks_matrix, threads, 0, stream>>>(
        d_dout,
        d_q,
        d_k,
        d_v,
        d_lse,
        d_drow,
        d_probs,
        d_dp,
        d_ds,
        rows,
        T,
        D,
        causal,
        softmax_scale);
    DenseAttentionDVKernel<<<blocks_vec, threads, 0, stream>>>(d_probs, d_dout, d_dv, rows, T, D);
    DenseAttentionDQKernel<<<blocks_vec, threads, 0, stream>>>(d_ds, d_k, d_dq, rows, T, D, softmax_scale);
    DenseAttentionDKKernel<<<blocks_vec, threads, 0, stream>>>(d_ds, d_q, d_dk, rows, T, D, softmax_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "DenseAttentionBackward Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaMemcpyAsync(dq, d_dq, size4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(dk, d_dk, size4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(dv, d_dv, size4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "DenseAttentionBackward Runtime Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}

void BenchmarkFlashAttention2Backward(
    float* dout,
    float* q,
    float* k,
    float* v,
    float* out,
    float* lse,
    int B,
    int H,
    int T,
    int D,
    int causal,
    float softmax_scale,
    int warmup,
    int repeats,
    float* avg_ms,
    unsigned long long* allocated_bytes) {
    size_t size4 = (size_t)B * H * T * D;
    size_t size3 = (size_t)B * H * T;

    int device = 0;
    cudaGetDevice(&device);
    if (g_fa2_device != -1 && g_fa2_device != device) {
      FreeFA2Buffers();
    }
    if (size4 > g_fa2_cap4 || size3 > g_fa2_cap3 || g_fa2_dout == nullptr) {
      FreeFA2Buffers();
      cudaMalloc(&g_fa2_dout, size4 * sizeof(float));
      cudaMalloc(&g_fa2_q, size4 * sizeof(float));
      cudaMalloc(&g_fa2_k, size4 * sizeof(float));
      cudaMalloc(&g_fa2_v, size4 * sizeof(float));
      cudaMalloc(&g_fa2_out, size4 * sizeof(float));
      cudaMalloc(&g_fa2_lse, size3 * sizeof(float));
      cudaMalloc(&g_fa2_dq, size4 * sizeof(float));
      cudaMalloc(&g_fa2_dk, size4 * sizeof(float));
      cudaMalloc(&g_fa2_dv, size4 * sizeof(float));
      g_fa2_cap4 = size4;
      g_fa2_cap3 = size3;
      g_fa2_device = device;
    }

    float *d_dout = g_fa2_dout, *d_q = g_fa2_q, *d_k = g_fa2_k, *d_v = g_fa2_v;
    float *d_out = g_fa2_out, *d_lse = g_fa2_lse;
    float *d_dq = g_fa2_dq, *d_dk = g_fa2_dk, *d_dv = g_fa2_dv;
    cudaStream_t stream = 0;

    cudaMemcpyAsync(d_dout, dout, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_q, q, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_k, k, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v, v, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out, out, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lse, lse, size3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int BQ = (D <= 64) ? 8 : 4;
    int BK = (D <= 64) ? 32 : 16;
    if (prop.major >= 8) {
      BK = (D <= 64) ? 64 : 32;
    } else if (prop.major == 7) {
      BK = (D <= 64) ? 32 : 16;
    } else {
      BK = (D <= 64) ? 16 : 8;
      BQ = (D <= 64) ? 4 : 2;
    }
    int threadsDQ = BQ * 32;
    if (threadsDQ > 256) {
      threadsDQ = 256;
    }
    int threadsDKDV = 256;
    int num_q_tiles = (T + BQ - 1) / BQ;
    int num_k_tiles = (T + BK - 1) / BK;
    int blocksDQ = B * H * num_q_tiles;
    int blocksDKDV = B * H * num_k_tiles;
    int D_PAD = D + 1;
    size_t sharedFloatsDQ = (size_t)(4 * BQ * D_PAD + 2 * BK * D_PAD + 4 * BQ);
    size_t baseSharedBytesDQ = sharedFloatsDQ * sizeof(float);
    size_t sharedFloatsDKDV = (size_t)(4 * BK * D_PAD + 3 * BQ * D_PAD + 2 * BQ + 2 * (threadsDKDV / 32) * BQ);
    size_t baseSharedBytesDKDV = sharedFloatsDKDV * sizeof(float);
    int use_tensor_core = 0;
    if (prop.major >= 7 && (D % 16 == 0) && D >= 128) {
      use_tensor_core = 1;
    }
    size_t tcSharedBytesDQ = 0;
    size_t tcSharedBytesDKDV = 0;
    if (use_tensor_core) {
      int dqWarps = threadsDQ / 32;
      int dkdvWarps = threadsDKDV / 32;
      tcSharedBytesDQ =
        (size_t)dqWarps * (16 * 16 * sizeof(half)) +
        (size_t)dqWarps * (16 * 16 * sizeof(half)) +
        (size_t)dqWarps * (16 * 16 * sizeof(float));
      tcSharedBytesDKDV =
        (size_t)dkdvWarps * (16 * 16 * sizeof(half)) +
        (size_t)dkdvWarps * (16 * 16 * sizeof(half)) +
        (size_t)dkdvWarps * (16 * 16 * sizeof(float));
    }
    size_t sharedBytesDQ = baseSharedBytesDQ + tcSharedBytesDQ;
    size_t sharedBytesDKDV = baseSharedBytesDKDV + tcSharedBytesDKDV;
    cudaFuncSetAttribute(
        FlashAttention2BackwardDQKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)sharedBytesDQ);
    cudaFuncSetAttribute(
        FlashAttention2BackwardDKDVKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)sharedBytesDKDV);

    for (int iter = 0; iter < warmup; ++iter) {
      FlashAttention2BackwardDQKernel<<<blocksDQ, threadsDQ, sharedBytesDQ, stream>>>(
          d_dout, d_q, d_k, d_v, d_out, d_lse, d_dq,
          B, H, T, D, BQ, BK, use_tensor_core, causal, softmax_scale);
      FlashAttention2BackwardDKDVKernel<<<blocksDKDV, threadsDKDV, sharedBytesDKDV, stream>>>(
          d_dout, d_q, d_k, d_v, d_out, d_lse, d_dk, d_dv,
          B, H, T, D, BQ, BK, use_tensor_core, causal, softmax_scale);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int iter = 0; iter < repeats; ++iter) {
      FlashAttention2BackwardDQKernel<<<blocksDQ, threadsDQ, sharedBytesDQ, stream>>>(
        d_dout, d_q, d_k, d_v, d_out, d_lse, d_dq,
          B, H, T, D, BQ, BK, use_tensor_core, causal, softmax_scale);
      FlashAttention2BackwardDKDVKernel<<<blocksDKDV, threadsDKDV, sharedBytesDKDV, stream>>>(
        d_dout, d_q, d_k, d_v, d_out, d_lse, d_dk, d_dv,
        B, H, T, D, BQ, BK, use_tensor_core, causal, softmax_scale);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "BenchmarkFlashAttention2Backward Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    *avg_ms = total_ms / ((repeats > 0) ? repeats : 1);
    *allocated_bytes = (unsigned long long)((8ULL * size4 + size3) * sizeof(float));
}

void BenchmarkDenseAttentionBackward(
    float* dout,
    float* q,
    float* k,
    float* v,
    float* out,
    float* lse,
    int B,
    int H,
    int T,
    int D,
    int causal,
    float softmax_scale,
    int warmup,
    int repeats,
    float* avg_ms,
    unsigned long long* allocated_bytes) {
    size_t size4 = (size_t)B * H * T * D;
    size_t size3 = (size_t)B * H * T;
    size_t size2 = (size_t)B * H * T * T;

    int device = 0;
    cudaGetDevice(&device);
    if (g_dense_device != -1 && g_dense_device != device) {
      FreeDenseBuffers();
    }
    if (size4 > g_dense_cap4 || size3 > g_dense_cap3 || size2 > g_dense_cap2 || g_dense_dout == nullptr) {
      FreeDenseBuffers();
      cudaMalloc(&g_dense_dout, size4 * sizeof(float));
      cudaMalloc(&g_dense_q, size4 * sizeof(float));
      cudaMalloc(&g_dense_k, size4 * sizeof(float));
      cudaMalloc(&g_dense_v, size4 * sizeof(float));
      cudaMalloc(&g_dense_out, size4 * sizeof(float));
      cudaMalloc(&g_dense_lse, size3 * sizeof(float));
      cudaMalloc(&g_dense_dq, size4 * sizeof(float));
      cudaMalloc(&g_dense_dk, size4 * sizeof(float));
      cudaMalloc(&g_dense_dv, size4 * sizeof(float));
      cudaMalloc(&g_dense_drow, size3 * sizeof(float));
      cudaMalloc(&g_dense_probs, size2 * sizeof(float));
      cudaMalloc(&g_dense_dp, size2 * sizeof(float));
      cudaMalloc(&g_dense_ds, size2 * sizeof(float));
      g_dense_cap4 = size4;
      g_dense_cap3 = size3;
      g_dense_cap2 = size2;
      g_dense_device = device;
    }

    float *d_dout = g_dense_dout, *d_q = g_dense_q, *d_k = g_dense_k, *d_v = g_dense_v;
    float *d_out = g_dense_out, *d_lse = g_dense_lse;
    float *d_dq = g_dense_dq, *d_dk = g_dense_dk, *d_dv = g_dense_dv;
    float *d_drow = g_dense_drow, *d_probs = g_dense_probs, *d_dp = g_dense_dp, *d_ds = g_dense_ds;
    cudaStream_t stream = 0;

    cudaMemcpyAsync(d_dout, dout, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_q, q, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_k, k, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v, v, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out, out, size4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lse, lse, size3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    int rows = B * H * T;
    int threads = 256;
    int blocks_rows = (rows + threads - 1) / threads;
    int blocks_matrix = ((int)(rows * T) + threads - 1) / threads;
    int blocks_vec = ((int)(rows * D) + threads - 1) / threads;

    for (int iter = 0; iter < warmup; ++iter) {
      DenseAttentionDRowKernel<<<blocks_rows, threads, 0, stream>>>(d_dout, d_out, d_drow, rows, D);
      DenseAttentionMaterializeKernel<<<blocks_matrix, threads, 0, stream>>>(
          d_dout, d_q, d_k, d_v, d_lse, d_drow, d_probs, d_dp, d_ds,
          rows, T, D, causal, softmax_scale);
      DenseAttentionDVKernel<<<blocks_vec, threads, 0, stream>>>(d_probs, d_dout, d_dv, rows, T, D);
      DenseAttentionDQKernel<<<blocks_vec, threads, 0, stream>>>(d_ds, d_k, d_dq, rows, T, D, softmax_scale);
      DenseAttentionDKKernel<<<blocks_vec, threads, 0, stream>>>(d_ds, d_q, d_dk, rows, T, D, softmax_scale);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int iter = 0; iter < repeats; ++iter) {
      DenseAttentionDRowKernel<<<blocks_rows, threads, 0, stream>>>(d_dout, d_out, d_drow, rows, D);
      DenseAttentionMaterializeKernel<<<blocks_matrix, threads, 0, stream>>>(
          d_dout, d_q, d_k, d_v, d_lse, d_drow, d_probs, d_dp, d_ds,
          rows, T, D, causal, softmax_scale);
      DenseAttentionDVKernel<<<blocks_vec, threads, 0, stream>>>(d_probs, d_dout, d_dv, rows, T, D);
      DenseAttentionDQKernel<<<blocks_vec, threads, 0, stream>>>(d_ds, d_k, d_dq, rows, T, D, softmax_scale);
      DenseAttentionDKKernel<<<blocks_vec, threads, 0, stream>>>(d_ds, d_q, d_dk, rows, T, D, softmax_scale);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "BenchmarkDenseAttentionBackward Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    *avg_ms = total_ms / ((repeats > 0) ? repeats : 1);
    *allocated_bytes = (unsigned long long)((8ULL * size4 + 2ULL * size3 + 3ULL * size2) * sizeof(float));
}

void MatrixMultiply(
    float* out,
    int* out_shape,
    int* out_strides,
    float* a_storage,
    int* a_shape,
    int* a_strides,
    float* b_storage,
    int* b_shape,
    int* b_strides,
    int batch, int m, int p
) {
    int n = a_shape[2];

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc(&d_a, batch * m * n * sizeof(float));
    cudaMalloc(&d_b, batch * n * p * sizeof(float));
    cudaMalloc(&d_out, batch * m * p * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, 3 * sizeof(int));
    cudaMalloc(&d_out_strides, 3 * sizeof(int));
    cudaMalloc(&d_a_shape, 3 * sizeof(int));
    cudaMalloc(&d_a_strides, 3 * sizeof(int));
    cudaMalloc(&d_b_shape, 3 * sizeof(int));
    cudaMalloc(&d_b_strides, 3 * sizeof(int));


    // Copy data to the device
    cudaMemcpy(d_a, a_storage, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, batch * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Matmul Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}

void tensorMap(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int in_size,
    int shape_size,
    int fn_id
) {

    float *d_out, *d_in;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_in, in_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_in_shape, shape_size * sizeof(int));
    cudaMalloc(&d_in_strides, shape_size * sizeof(int));

    cudaMemcpy(d_in, in_storage, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, 
      d_in, d_in_shape, d_in_strides, 
      shape_size, fn_id);
    
    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Map Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
}


void tensorZip(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    float* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_size,
    int a_shape_size,
    float* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_size,
    int b_shape_size,
    int fn_id
) {

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc((void **)&d_a, a_size * sizeof(float));
    cudaMalloc(&d_b, b_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, out_shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, out_shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, a_shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, a_shape_size * sizeof(int));
    cudaMalloc(&d_b_shape, b_shape_size * sizeof(int));
    cudaMalloc(&d_b_strides, b_shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, out_shape_size,
      d_a, d_a_shape, d_a_strides, a_shape_size,
      d_b, d_b_shape, d_b_strides, b_shape_size,
      fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();


    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Zip Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}



void tensorReduce(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim, 
    float reduce_value,
    int shape_size,
    int fn_id
) {
    int a_size = out_size * a_shape[reduce_dim];
    float *d_out, *d_a;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_a, a_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, shape_size * sizeof(int));

    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size, 
        d_a, d_a_shape, d_a_strides, 
        reduce_dim, reduce_value, shape_size, fn_id
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Reduce Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
}

}