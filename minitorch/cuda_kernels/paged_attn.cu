/*
 * paged_attn.cu: CUDA attention kernel for paged KV cache (decode step only).
 *
 * One CUDA block handles one (batch, head) pair.  Threads within the block
 * collaborate on the dot-product reduction and the online-softmax weighted
 * accumulation for every token in that sequence's history.
 *
 * Grid  : B * n_head blocks
 * Block : BLOCK_SIZE threads  (must be a power of 2; 64 covers head_dim <= 128
 *         with at most two passes through the per-thread loop)
 *
 * Shared memory layout per block (dynamic, allocated at launch):
 *   [0             .. BLOCK_SIZE)    dot-product partial sums
 *   [BLOCK_SIZE    .. BLOCK_SIZE+D)  output accumulator   (D = head_dim)
 *   [BLOCK_SIZE+D  .. BLOCK_SIZE+2D) cached query vector
 *
 * KV layout received by the kernel (single-layer slice, row-major):
 *   kv_k / kv_v : [n_blocks][n_head][block_size][head_dim]
 *
 * The extern "C" launcher copies only the current layer's slice of the block
 * pool to the GPU, runs the kernel, then copies the result back.  The rest of
 * the block pool (all other layers) is never transferred.
 *
 * Build:
 *   nvcc -O2 -shared -Xcompiler -fPIC \
 *        -o minitorch/cuda_kernels/paged_attn.so \
 *        minitorch/cuda_kernels/paged_attn.cu
 */

#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 64

__global__ void pagedAttentionKernel(
    float*       output,          /* (B, n_head, head_dim)                       */
    const float* queries,         /* (B, n_head, head_dim)                       */
    const float* kv_k_layer,      /* (n_blocks, n_head, block_size, head_dim)    */
    const float* kv_v_layer,      /* same shape                                  */
    const int*   block_tables,    /* (B, max_blocks_per_seq)  -1 = unused        */
    const int*   seq_lengths,     /* (B,)                                        */
    int B, int n_head, int head_dim,
    int block_size, int max_blocks_per_seq,
    float scale
) {
    extern __shared__ float shmem[];
    float* dot_partial = shmem;                      /* [BLOCK_SIZE]  */
    float* acc         = shmem + BLOCK_SIZE;         /* [head_dim]    */
    float* q_cache     = shmem + BLOCK_SIZE + head_dim; /* [head_dim] */

    int bh  = blockIdx.x;
    int b   = bh / n_head;
    int h   = bh % n_head;
    int tid = threadIdx.x;

    if (b >= B) return;

    long bh_stride     = (long)n_head     * head_dim;
    long block_stride  = (long)n_head     * block_size * head_dim;
    long head_stride_k = (long)block_size * head_dim;

    const float* q = queries + (long)b * bh_stride + (long)h * head_dim;
    float*       o = output  + (long)b * bh_stride + (long)h * head_dim;

    /* cache query and zero accumulator */
    for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
        q_cache[i] = q[i];
        acc[i]     = 0.0f;
    }
    dot_partial[tid] = 0.0f;
    __syncthreads();

    float m = -3.402823466e+38f;
    float d = 0.0f;

    int total = seq_lengths[b];

    for (int t = 0; t < total; t++) {
        int phys = block_tables[(long)b * max_blocks_per_seq + t / block_size];
        int slot = t % block_size;

        const float* k = kv_k_layer
            + (long)phys * block_stride
            + (long)h    * head_stride_k
            + (long)slot * head_dim;
        const float* v = kv_v_layer
            + (long)phys * block_stride
            + (long)h    * head_stride_k
            + (long)slot * head_dim;

        /* parallel dot product: each thread accumulates a partial sum */
        float partial = 0.0f;
        for (int i = tid; i < head_dim; i += BLOCK_SIZE)
            partial += q_cache[i] * k[i];
        dot_partial[tid] = partial;
        __syncthreads();

        /* tree reduction to dot_partial[0] */
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s)
                dot_partial[tid] += dot_partial[tid + s];
            __syncthreads();
        }
        float score = dot_partial[0] * scale;
        /* broadcast: all threads read the same score from shared mem */
        __syncthreads();

        /* online softmax update, identical across all threads in the block */
        float m_new     = fmaxf(score, m);
        float exp_shift = expf(m - m_new);
        float exp_score = expf(score - m_new);
        d = d * exp_shift + exp_score;
        m = m_new;

        /* update output accumulator in parallel */
        for (int i = tid; i < head_dim; i += BLOCK_SIZE)
            acc[i] = acc[i] * exp_shift + v[i] * exp_score;
        __syncthreads();
    }

    /* normalize and write output */
    float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
    for (int i = tid; i < head_dim; i += BLOCK_SIZE)
        o[i] = acc[i] * inv_d;
}

extern "C" void pagedAttention(
    float*       output,
    const float* queries,
    const float* kv_k,
    const float* kv_v,
    const int*   block_tables,
    const int*   seq_lengths,
    int layer_idx,
    int B, int n_head, int head_dim,
    int block_size, int n_blocks,
    int max_blocks_per_seq, float scale
) {
    long layer_stride  = (long)n_blocks * n_head * block_size * head_dim;
    size_t layer_bytes = (size_t)n_blocks * n_head * block_size * head_dim * sizeof(float);
    size_t q_bytes     = (size_t)B * n_head * head_dim * sizeof(float);
    size_t bt_bytes    = (size_t)B * max_blocks_per_seq * sizeof(int);
    size_t sl_bytes    = (size_t)B * sizeof(int);

    float *d_q, *d_kv_k, *d_kv_v, *d_out;
    int   *d_bt, *d_sl;

    cudaMalloc(&d_q,    q_bytes);
    cudaMalloc(&d_kv_k, layer_bytes);
    cudaMalloc(&d_kv_v, layer_bytes);
    cudaMalloc(&d_bt,   bt_bytes);
    cudaMalloc(&d_sl,   sl_bytes);
    cudaMalloc(&d_out,  q_bytes);

    cudaMemcpy(d_q,    queries,                                  q_bytes,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_kv_k, kv_k + (long)layer_idx * layer_stride,   layer_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kv_v, kv_v + (long)layer_idx * layer_stride,   layer_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt,   block_tables,                             bt_bytes,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_sl,   seq_lengths,                              sl_bytes,    cudaMemcpyHostToDevice);

    int    grid_size   = B * n_head;
    size_t shmem_bytes = (size_t)(BLOCK_SIZE + 2 * head_dim) * sizeof(float);

    pagedAttentionKernel<<<grid_size, BLOCK_SIZE, shmem_bytes>>>(
        d_out, d_q, d_kv_k, d_kv_v,
        d_bt, d_sl,
        B, n_head, head_dim, block_size, max_blocks_per_seq, scale
    );

    /* cudaMemcpyDeviceToHost blocks until the kernel finishes */
    cudaMemcpy(output, d_out, q_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_kv_k);
    cudaFree(d_kv_v);
    cudaFree(d_bt);
    cudaFree(d_sl);
    cudaFree(d_out);
}
