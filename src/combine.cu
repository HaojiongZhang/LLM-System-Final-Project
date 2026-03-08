#include <cuda_runtime.h>
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


__global__ void FlashAttention2BackwardKernel(
    const float* dout,
    const float* q,
    const float* k,
    const float* v,
    const float* out,
    const float* lse,
    float* dq,
    float* dk,
    float* dv,
    int B,
    int H,
    int T,
    int D,
    int causal,
    float softmax_scale)
{
  // Logical input/output shapes:
  // q,k,v,out,dout,dq,dk,dv: (B,H,T,D)
  // lse:                     (B,H,T)
  // This kernel maps one thread -> one attention pair (i,j) for fixed (b,h).
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * T * T;
  if (idx >= total) {
    return;
  }

  // Decode flattened index into tuple (b,h,i,j).
  int j = idx % T;
  int rem = idx / T;
  int i = rem % T;
  rem /= T;
  int h = rem % H;
  int b = rem / H;

  // Causal mask rule: query i cannot attend to key j when j > i.
  if (causal && j > i) {
    return;
  }

  // Flat row/col offsets into compact (B,H,T,D) layout.
  int row4 = ((b * H + h) * T + i) * D;
  int col4 = ((b * H + h) * T + j) * D;
  // Offset into compact (B,H,T) layout for row i.
  int row3 = ((b * H + h) * T + i);

  // score_ij = softmax_scale * dot(q_i, k_j)
  float score = 0.0f;
  for (int d = 0; d < D; ++d) {
    score += q[row4 + d] * k[col4 + d];
  }
  score *= softmax_scale;

  // p_ij = exp(score_ij - lse_i)
  float p = expf(score - lse[row3]);

  // dP_ij = dot(dO_i, V_j)
  // D_i   = dot(dO_i, O_i)
  // dS_ij = P_ij * (dP_ij - D_i)
  float dP = 0.0f;
  float Drow = 0.0f;
  for (int d = 0; d < D; ++d) {
    dP += dout[row4 + d] * v[col4 + d];
    Drow += dout[row4 + d] * out[row4 + d];
  }

  float dS = p * (dP - Drow);

  // Accumulate gradients:
  // dV_j += P_ij * dO_i
  // dQ_i += dS_ij * K_j * scale
  // dK_j += dS_ij * Q_i * scale
  // Multiple (i,j) pairs may update same row/col -> atomicAdd required.
  for (int d = 0; d < D; ++d) {
    atomicAdd(&dv[col4 + d], p * dout[row4 + d]);
    atomicAdd(&dq[row4 + d], dS * k[col4 + d] * softmax_scale);
    atomicAdd(&dk[col4 + d], dS * q[row4 + d] * softmax_scale);
  }
}


extern "C" {

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

    float *d_dout, *d_q, *d_k, *d_v, *d_out, *d_lse;
    float *d_dq, *d_dk, *d_dv;

    // Device allocations for all inputs/outputs.
    cudaMalloc(&d_dout, size4 * sizeof(float));
    cudaMalloc(&d_q, size4 * sizeof(float));
    cudaMalloc(&d_k, size4 * sizeof(float));
    cudaMalloc(&d_v, size4 * sizeof(float));
    cudaMalloc(&d_out, size4 * sizeof(float));
    cudaMalloc(&d_lse, size3 * sizeof(float));
    cudaMalloc(&d_dq, size4 * sizeof(float));
    cudaMalloc(&d_dk, size4 * sizeof(float));
    cudaMalloc(&d_dv, size4 * sizeof(float));

    // Copy host tensors to device.
    cudaMemcpy(d_dout, dout, size4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, size4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lse, lse, size3 * sizeof(float), cudaMemcpyHostToDevice);

    // Zero output gradient buffers before atomic accumulation in kernel.
    cudaMemset(d_dq, 0, size4 * sizeof(float));
    cudaMemset(d_dk, 0, size4 * sizeof(float));
    cudaMemset(d_dv, 0, size4 * sizeof(float));

    // One thread per (b,h,i,j) pair.
    int total = B * H * T * T;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    FlashAttention2BackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_dout,
        d_q,
        d_k,
        d_v,
        d_out,
        d_lse,
        d_dq,
        d_dk,
        d_dv,
        B,
        H,
        T,
        D,
        causal,
        softmax_scale);

    // Synchronize so any launch/runtime errors become visible here.
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "FlashAttention2Backward Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Copy gradients back to host buffers expected by ctypes caller.
    cudaMemcpy(dq, d_dq, size4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dk, d_dk, size4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dv, d_dv, size4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Release all temporary device allocations.
    cudaFree(d_dout);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_lse);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dv);
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