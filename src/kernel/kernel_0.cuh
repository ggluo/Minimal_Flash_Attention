#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// =========================================================================
// Configuration & Constants
// =========================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32


// =========================================================================
// Device Helper: Load float -> half shared memory
// =========================================================================
__device__ void load_float_to_half(half *shmem_ptr, const float *global_ptr, 
                                   int rows, int cols, int stride, 
                                   int row_offset, int col_offset, 
                                   int max_rows, int max_cols) 
{
    // Each warp loads a 16x16 tile (256 elements)
    // 32 threads -> 8 elements per thread
    int tid = threadIdx.x;
    int lane_row = tid / 2; // Map threads to coverage
    
    // Simple coalesced loading strategy for 16x16 block
    // Thread i loads indices i, i+32, i+64...
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = tid + i * 32;
        int r = idx / 16;
        int c = idx % 16;
        
        int global_r = row_offset + r;
        int global_c = col_offset + c;

        float val = 0.0f;
        if (global_r < max_rows && global_c < max_cols) {
            val = global_ptr[global_r * stride + global_c];
        }
        shmem_ptr[r * 16 + c] = __float2half(val);
    }
}

// =========================================================================
// Kernel: FlashAttention WMMA
// 
// Grid: (N / 16, d / 16)
// Block: 32 threads (1 warp)
// Shared Mem: Requires space for Q(16x16), K(16x16), V(16x16) + buffers
// =========================================================================
__global__ void attention_wmma(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    int N, int d,
    float *__restrict__ O)
{
    // ----------------------------------------------------------------
    // 1. Setup & Shared Memory Allocation
    // ----------------------------------------------------------------
    // We need segments for Q, K, V (in half) and O/Scores (in float/half)
    // Layout:
    // [0]: Q_tile (16x16 half) = 256 * 2 bytes
    // [1]: K_tile (16x16 half) = 256 * 2 bytes
    // [2]: V_tile (16x16 half) = 256 * 2 bytes
    // [3]: Temp/Accumulator buffer (16x16 float) = 256 * 4 bytes
    extern __shared__ half shmem_pool[];
    
    half *shmem_q = shmem_pool;
    half *shmem_k = shmem_q + 256;
    half *shmem_v = shmem_k + 256;
    float *shmem_f = (float*)&shmem_v[256]; // Reusable float buffer

    // Indices
    int tx = threadIdx.x;
    int q_block_idx = blockIdx.x; // Which chunk of Queries (rows 0..15, 16..31)
    int head_block_idx = blockIdx.y; // Which chunk of Head Dim (cols 0..15, 16..31)
    
    int row_q_start = q_block_idx * 16;
    int col_d_start = head_block_idx * 16;

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag; // Transposed load for K
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
    
    // Accumulators
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s; // Scores
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o; // Output

    wmma::fill_fragment(acc_o, 0.0f);

    // Online Softmax Stats
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float softmax_scale = 1.0f / sqrtf((float)d);

    // ----------------------------------------------------------------
    // 2. Loop over Sequence Length (N) in chunks of 16 (The 'K' and 'V' blocks)
    // ----------------------------------------------------------------
    for (int k_block = 0; k_block < N; k_block += 16) {
        
        // --- Step A: Compute S = Q * K^T ---
        // Q is 16xd, K is 16xd (transposed conceptually to dx16 for mult)
        // We accumulate over the 'd' dimension.
        wmma::fill_fragment(acc_s, 0.0f);

        for (int d_chunk = 0; d_chunk < d; d_chunk += 16) {
            // Load Q sub-tile (16x16)
            load_float_to_half(shmem_q, Q, 16, 16, d, row_q_start, d_chunk, N, d);
            
            // Load K sub-tile (16x16). Note: K is N x d.
            // We load standard layout, but interpret as col_major in MMA or transpose here.
            // wmma::col_major expects the data in memory to be column major. 
            // Our data in Global/Shmem is Row Major. 
            // To get K^T, we rely on the fragment loader. 
            // Loading (16x16) row-major data into a col_major fragment effectively transposes it 
            // IF the matrix was square? No.
            // We need Q(16x16) * K^T(16x16). 
            // If we load K(row major) into Matrix_B(col_major), it treats rows as cols.
            // So K(i,j) becomes B(j,i). Correct for K^T.
            load_float_to_half(shmem_k, K, 16, 16, d, k_block, d_chunk, N, d);
            
            __syncthreads(); // Wait for loads
            
            wmma::load_matrix_sync(q_frag, shmem_q, 16);
            wmma::load_matrix_sync(k_frag, shmem_k, 16);
            
            wmma::mma_sync(acc_s, q_frag, k_frag, acc_s);
            __syncthreads(); // Sync for safety before next load
        }

        // --- Step B: Online Softmax Correction ---
        
        // 1. Store Scores to Shared Memory (float)
        wmma::store_matrix_sync(shmem_f, acc_s, 16, wmma::mem_row_major);
        __syncthreads();

        // 2. Compute Max/Exp/Sum per row
        //    Since we have 32 threads and 16 rows, threads 0-15 handle rows 0-15.
        float m_curr = -INFINITY;
        float l_curr = 0.0f;
        float correction = 1.0f; // Default if no update
        
        // Parallel Softmax (Naive per-thread row handling)
        if (tx < 16) {
            int row = tx;
            float row_max = -INFINITY;
            
            // Find max
            for (int c = 0; c < 16; ++c) {
                // Apply Masking here if needed (e.g. causal)
                // For now: pure attention
                float val = shmem_f[row * 16 + c] * softmax_scale;
                shmem_f[row * 16 + c] = val; // Scale in place
                if (val > row_max) row_max = val;
            }
            
            // Update Max Stats
            // m_prev is thread-local register containing stats for this row
            float m_old = m_prev;
            m_curr = fmaxf(m_old, row_max);
            
            // Compute Exp & Sum
            float row_sum = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float val = __expf(shmem_f[row * 16 + c] - m_curr);
                // Store P (prob) in half precision buffer (reuse shmem_q or shmem_k)
                // We reuse shmem_q for P.
                shmem_q[row * 16 + c] = __float2half(val); 
                row_sum += val;
            }
            
            // Update Sum Stats
            // O_new = O_old * exp(m_old - m_curr) + P * V
            correction = __expf(m_old - m_curr);
            l_prev = l_prev * correction + row_sum;
            m_prev = m_curr;
            
            // Store correction factor to a known location for the whole warp to use?
            // Actually, we need to rescale the ACCUMULATOR 'acc_o'.
            // Since we can't easily map thread-to-accumulator-element, 
            // we will scale acc_o via shared memory.
            
            // Store correction in a buffer. Reuse shmem_v part or shmem_f end.
            shmem_f[256 + row] = correction; 
        }
        __syncthreads();

        // --- Step C: Rescale Output Accumulator ---
        
        // Store O to Shared Memory
        wmma::store_matrix_sync(shmem_f, acc_o, 16, wmma::mem_row_major);
        __syncthreads();

        // Apply Correction to O (Parallel over 256 elements)
        // Correction factor depends on ROW.
        for (int i = tx; i < 256; i += 32) {
            int r = i / 16;
            float corr = shmem_f[256 + r];
            shmem_f[i] *= corr;
        }
        __syncthreads();

        // Load O back
        wmma::load_matrix_sync(acc_o, shmem_f, 16, wmma::mem_row_major);
        
        // --- Step D: Compute O += P * V ---
        
        // Load P (calculated in Step B) into fragment
        wmma::load_matrix_sync(p_frag, shmem_q, 16);
        
        // Load V sub-tile (16x16)
        // We need the V block corresponding to the current k_block (which is row index for V)
        // and the current head_dim block (col index for V/O).
        load_float_to_half(shmem_v, V, 16, 16, d, k_block, col_d_start, N, d);
        __syncthreads();
        
        wmma::load_matrix_sync(v_frag, shmem_v, 16);
        
        // Accumulate
        wmma::mma_sync(acc_o, p_frag, v_frag, acc_o);
        __syncthreads();
    }

    // ----------------------------------------------------------------
    // 3. Finalize and Store Output
    // ----------------------------------------------------------------
    
    // Store O accumulator to Shared Memory
    wmma::store_matrix_sync(shmem_f, acc_o, 16, wmma::mem_row_major);
    __syncthreads();
    
    // Normalize by l_prev and write to Global
    if (tx < 16) {
        int r = tx;
        float div = 1.0f / (l_prev + 1e-8f); // Safety epsilon
        
        for (int c = 0; c < 16; ++c) {
            float val = shmem_f[r * 16 + c];
            val *= div;
            
            int global_r = row_q_start + r;
            int global_c = col_d_start + c;
            
            if (global_r < N && global_c < d) {
                O[global_r * d + global_c] = val;
            }
        }
    }
}