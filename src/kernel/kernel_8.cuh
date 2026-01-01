#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// =========================================================================
// Helper: FP32 -> FP16 conversion
// =========================================================================
__device__ inline __half to_half(float f) {
    return __float2half(f);
}

// =========================================================================
// Kernel: FlashAttention using WMMA (Tensor Cores)
// Block Size: 32 threads (1 Warp)
// Grid Size: (ceil(d / 16), ceil(N / 16))
// Each block computes a 16x16 output tile (16 seq rows x 16 head cols)
// Now supports arbitrary d (tiles inner dot product dimension)
// Fixed partial dot product issue for d > 16
// =========================================================================
__global__ void attention_wmma(const float *__restrict__ inputQ,
                               const float *__restrict__ inputK,
                               const float *__restrict__ inputV,
                               int N, int d,
                               float *__restrict__ output)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int row_offset = blockIdx.y * WMMA_M;
    int col_offset = blockIdx.x * WMMA_N;
    float scale = 1.0f / sqrtf((float)d);

    if (row_offset >= N || col_offset >= d) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_Q;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_K;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_V;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_P;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_S;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_O;

    wmma::fill_fragment(acc_O, 0.0f);

    __shared__ __half smem_buffer[WMMA_M * WMMA_K];
    __shared__ float smem_S[WMMA_M * WMMA_N];
    __shared__ float smem_O[WMMA_M * WMMA_N];
    __shared__ float smem_stats_max[WMMA_M];
    __shared__ float smem_stats_sum[WMMA_M];
    __shared__ float smem_factor[WMMA_M];

    int laneId = threadIdx.x;

    if (threadIdx.x < WMMA_M) {
        smem_stats_max[threadIdx.x] = -1e20f;
        smem_stats_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int k_idx = 0; k_idx < N; k_idx += WMMA_K) {
        // === Inner tiling over head dimension d for full Q @ K^T ===
        wmma::fill_fragment(acc_S, 0.0f);

        for (int dot_k = 0; dot_k < d; dot_k += WMMA_K) {
            // Load Q tile (16 rows x 16 dot)
            for (int i = laneId; i < WMMA_M * WMMA_K; i += 32) {
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                float val = 0.0f;
                if ((row_offset + r < N) && (dot_k + c < d)) {
                    val = inputQ[(row_offset + r) * d + (dot_k + c)];
                }
                smem_buffer[i] = to_half(val);
            }
            __syncthreads();
            wmma::load_matrix_sync(frag_Q, smem_buffer, WMMA_K);

            // Load K tile (16 seq x 16 dot)
            for (int i = laneId; i < WMMA_M * WMMA_K; i += 32) {
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                float val = 0.0f;
                if ((k_idx + r < N) && (dot_k + c < d)) {
                    val = inputK[(k_idx + r) * d + (dot_k + c)];
                }
                smem_buffer[i] = to_half(val);
            }
            __syncthreads();
            wmma::load_matrix_sync(frag_K, smem_buffer, WMMA_K);  // col_major → effective transpose

            // Accumulate partial S
            wmma::mma_sync(acc_S, frag_Q, frag_K, acc_S);
        }

        // Now acc_S contains full partial scores for this seq block (16 x 16)
        wmma::store_matrix_sync(smem_S, acc_S, WMMA_N, wmma::mem_row_major);
        for (int i = laneId; i < WMMA_M * WMMA_N; i += 32) {
            smem_S[i] *= scale;
        }
        __syncthreads();

        // Online softmax
        if (threadIdx.x < WMMA_M) {
            int row = threadIdx.x;
            float local_max = -1e20f;
            for (int c = 0; c < WMMA_N; c++) {
                local_max = fmaxf(local_max, smem_S[row * WMMA_N + c]);
            }
            float old_max = smem_stats_max[row];
            float new_max = fmaxf(old_max, local_max);
            float factor = expf(old_max - new_max);
            smem_factor[row] = factor;
            smem_stats_max[row] = new_max;
            smem_stats_sum[row] *= factor;
            for (int c = 0; c < WMMA_N; c++) {
                smem_stats_sum[row] += expf(smem_S[row * WMMA_N + c] - new_max);
            }
        }
        __syncthreads();

        // Rescale previous acc_O
        wmma::store_matrix_sync(smem_O, acc_O, WMMA_N, wmma::mem_row_major);
        __syncthreads();
        for (int i = laneId; i < WMMA_M * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            smem_O[i] *= smem_factor[row];
        }
        __syncthreads();
        wmma::load_matrix_sync(acc_O, smem_O, WMMA_N, wmma::mem_row_major);

        // Compute P = exp(S - new_max)
        for (int i = laneId; i < WMMA_M * WMMA_N; i += 32) {
            int rr = i / WMMA_N;
            int cc = i % WMMA_N;
            float val = expf(smem_S[rr * WMMA_N + cc] - smem_stats_max[rr]);
            smem_buffer[i] = to_half(val);
        }
        __syncthreads();
        wmma::load_matrix_sync(frag_P, smem_buffer, WMMA_N);

        // Load V tile (16 seq x 16 output head cols) — no inner tiling needed here
        for (int i = laneId; i < WMMA_M * WMMA_K; i += 32) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            float val = 0.0f;
            if ((k_idx + r < N) && (col_offset + c < d)) {
                val = inputV[(k_idx + r) * d + (col_offset + c)];
            }
            smem_buffer[i] = to_half(val);
        }
        __syncthreads();
        wmma::load_matrix_sync(frag_V, smem_buffer, WMMA_K);

        // acc_O += P * V (partial for this seq block)
        wmma::mma_sync(acc_O, frag_P, frag_V, acc_O);
    }

    // Final normalize and store
    wmma::store_matrix_sync(smem_O, acc_O, WMMA_N, wmma::mem_row_major);
    for (int i = laneId; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N;
        int c = i % WMMA_N;
        if ((row_offset + r < N) && (col_offset + c < d)) {
            float val = smem_O[r * WMMA_N + c];
            val /= (smem_stats_sum[r] + 1e-6f);
            output[(row_offset + r) * d + (col_offset + c)] = val;
        }
    }
}