/*
 * paged_attn.c: CPU-only fallback for paged KV cache attention.
 *
 * Superseded by paged_attn.cu which runs the same algorithm on the GPU.
 * Keep this file for CPU-only environments where nvcc is unavailable.
 *
 * Each call handles a batch of single-token queries attending over their
 * full KV history stored in a non-contiguous block pool.  Online softmax
 * is used so no O(T) score buffer is allocated.
 *
 * KV layout (row-major):
 *   kv_k / kv_v : [n_layers][n_blocks][n_head][block_size][head_dim]
 *
 * Build (CPU fallback only):
 *   gcc -O2 -shared -fPIC -o minitorch/cuda_kernels/paged_attn.so \
 *       minitorch/cuda_kernels/paged_attn.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

void pagedAttention(
    float*       output,        /* (B, n_head, head_dim)                         */
    const float* queries,       /* (B, n_head, head_dim)                         */
    const float* kv_k,          /* (n_layers, n_blocks, n_head, block_size, head_dim) */
    const float* kv_v,          /* same shape as kv_k                            */
    const int*   block_tables,  /* (B, max_blocks_per_seq)  -1 = unused slot     */
    const int*   seq_lengths,   /* (B,)                                          */
    int layer_idx,
    int B,
    int n_head,
    int head_dim,
    int block_size,
    int n_blocks,
    int max_blocks_per_seq,
    float scale
)
{
    long layer_stride  = (long)n_blocks   * n_head * block_size * head_dim;
    long block_stride  = (long)n_head     * block_size * head_dim;
    long head_stride_k = (long)block_size * head_dim;

    long q_b_stride = (long)n_head * head_dim;
    long q_h_stride = (long)head_dim;

    float* o_acc = (float*)malloc((size_t)head_dim * sizeof(float));

    for (int b = 0; b < B; b++) {
        int total_tokens = seq_lengths[b];

        for (int h = 0; h < n_head; h++) {
            const float* q = queries + b * q_b_stride + h * q_h_stride;

            /* online softmax state */
            float m = -3.402823466e+38f;
            float d = 0.0f;
            memset(o_acc, 0, (size_t)head_dim * sizeof(float));

            for (int t  = 0; t < total_tokens; t++) {
                int block_idx  = t / block_size;
                int slot_idx   = t % block_size;
                int phys_block = block_tables[b * max_blocks_per_seq + block_idx];

                const float* k = kv_k
                    + (long)layer_idx  * layer_stride
                    + (long)phys_block * block_stride
                    + (long)h          * head_stride_k
                    + (long)slot_idx   * head_dim;

                const float* v = kv_v
                    + (long)layer_idx  * layer_stride
                    + (long)phys_block * block_stride
                    + (long)h          * head_stride_k
                    + (long)slot_idx   * head_dim;

                float score = 0.0f;
                for (int i = 0; i < head_dim; i++)
                    score += q[i] * k[i];
                score *= scale;

                float m_new     = (score > m) ? score : m;
                float exp_shift = expf(m - m_new);
                float exp_score = expf(score - m_new);

                d = d * exp_shift + exp_score;
                for (int i = 0; i < head_dim; i++)
                    o_acc[i] = o_acc[i] * exp_shift + v[i] * exp_score;

                m = m_new;
            }

            float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
            float* out_ptr = output + b * q_b_stride + h * q_h_stride;
            for (int i = 0; i < head_dim; i++)
                out_ptr[i] = o_acc[i] * inv_d;
        }
    }

    free(o_acc);
}
