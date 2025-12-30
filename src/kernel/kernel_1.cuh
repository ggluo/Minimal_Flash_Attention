__global__ void attention_v1(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d, int Br, int Bc,
                                 float *__restrict output)
{
    // Q,K,V are Nxd matrices, O = softmax(Q * K^T, dim=-1) * V
    // each thread handle one element in the output matrix O[row, col]
    // to compute O[row, col], we need to iterate over the full K matrix 
    // through the dimention N, and access row vector Q[row, :] and col vector V[:, col]
    // we adopt tile based parallelism where block size is Br x Bc
    // Br is for the rows in Q, Bc is for the columns in V
    // this means we need to access Br rows of Q[row:row+Br, :] and Bc columns of V[:, col:col+Bc]
    // So we have a grid of (N/Br) x (d/Bc), and the global index of a thread is
    int row = threadIdx.x + blockIdx.x * blockDim.x; // row index
    int col = threadIdx.y + blockIdx.y * blockDim.y; // col index
    if (row >= N || col >= d) return;

    // We choose Bc as step size when looping over K and V through the dimention N
    int steps = (N + Bc - 1) / Bc;

    // shared memory from extern
    extern __shared__ float sram[];

    // declare pointers to shared memory
    // QK_T is a Br x Bc matrix, dot product of Q[row:row+Br, :] and K_T[:, col:col+Bc]
    // SV is a Br x Bc matrix, online softmax(QK.T)V is a Br x Bc matrix
    float *b_sum = sram;
    float *b_max = sram + Br * Bc;
    float *QK_T = sram + Br * Bc * 2;
    float *SV = sram + Br * Bc * 3;

    float curr_max = -__FLT_MAX__;
    float prev_max = -__FLT_MAX__;
    float curr_sum = 1.0f;
    float prev_sum = 1.0f;

    // O[row, col]
    float acc_o = 0.0f;
    // loop over K and V using step size Bc
    for (int j = 0; j < steps; j++)
    {
        // 1D index for first column in the block, where threadIdx.y=0
        int start = threadIdx.x * Bc;
        // 2D index for block
        int block_index = start + threadIdx.y; 

        // reset
        SV[block_index] = 0.0f;

        // 1D index for the j-th selected segment in the first row of K matrix, K[j * Bc:j * Bc + Bc, :]
        int k_row_index = j * Bc + threadIdx.y;
        float qk_dot = 0.0f;
        for (int index = 0; index < d; index++)
        {
            qk_dot += inputQ[row * d + index] * inputK[k_row_index * d + index];
        }
        if (row < N && k_row_index < N)
        {
            b_max[block_index] = qk_dot; 
            QK_T[block_index]  = qk_dot;
            b_sum[block_index] = 1.0f;
        }
        else
        {
            QK_T[block_index]  = 0.0f;
            b_max[block_index] = -__FLT_MAX__;
            b_sum[block_index] = 0.0f;
        }
        __syncthreads();

        // perform 2D reduction
        for (int strip = Bc / 2; strip > 0; strip /= 2)
        {
            if (threadIdx.y < strip) // every time strip is halved, only the half of previous threads are active
            {
                if (b_max[block_index] > b_max[block_index + strip])
                {
                    b_sum[block_index] = b_sum[block_index] + b_sum[block_index + strip] *
                            __expf(b_max[block_index + strip] - b_max[block_index]);
                }
                else
                {
                    b_sum[block_index] = b_sum[block_index + strip] + b_sum[block_index] *
                            __expf(b_max[block_index] - b_max[block_index + strip]);
                    b_max[block_index] = b_max[block_index + strip];
                }
            }
            __syncthreads();
        }

        if (curr_max > b_max[start]) 
        {
            prev_sum = curr_sum;
            prev_max = curr_max;
            curr_sum = prev_sum + b_sum[start] *__expf(b_max[start] - curr_max);
        }
        else
        {
            prev_sum = curr_sum;
            prev_max = curr_max;
            curr_max = b_max[start];
            curr_sum = prev_sum * __expf(prev_max - curr_max) + b_sum[start];
        }

        __syncthreads();

        for (int phc = 0; phc < Bc; phc++)
        {
            if (phc + j * Bc < N)
            {
                SV[block_index] += __expf(QK_T[start + phc] - curr_max) * inputV[(phc + j * Bc) * d + col];
            }
        }
        acc_o = __expf(prev_max - curr_max) * acc_o * prev_sum + SV[block_index];
        acc_o = acc_o * __fdividef(1.0F, curr_sum);
        __syncthreads();
    }
    output[row * d + col] = acc_o;
}


__global__ void test(const float *__restrict__ inputQ,
                             const float *__restrict__ inputK,
                             const float *__restrict__ inputV,
                             int N, int d, int Br, int Bc,
                             float *__restrict__ output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Br rows per block
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // Bc columns per block

    if (row >= N || col >= d) return;

    int steps = (N + Bc - 1) / Bc;

    extern __shared__ float sram[];
    float *QK_T  = sram;                  // Br × Bc
    float *local_max = sram + Br * Bc;    // Br
    float *local_sum = sram + Br * Bc + Br; // Br

    // Per-thread registers for online softmax (one per output row)
    float thread_max = -__FLT_MAX__;
    float thread_sum = 0.0f;
    float thread_out = 0.0f;  // accumulated o'

    for (int tile = 0; tile < steps; ++tile) {
        int k_start = tile * Bc;

        // ---- 1. Compute Q[row] · K[k_start + threadIdx.y]  (cooperative over y) ----
        float qk = 0.0f;
        int k_idx = k_start + threadIdx.y;
        if (k_idx < N) {
            for (int i = 0; i < d; ++i) {
                qk += inputQ[row * d + i] * inputK[k_idx * d + i];
            }
        } else {
            qk = -__FLT_MAX__;
        }

        // Store the full row of logits for this tile (one value per thread in y)
        int idx = threadIdx.x * Bc + threadIdx.y;
        QK_T[idx] = qk;

        __syncthreads();

        // ---- 2. Per-row reduction in the block to get local max and exp-sum ----
        // Only need to do this once per row → let threadIdx.y == 0 handle it
        if (threadIdx.y == 0) {
            float lmax = -__FLT_MAX__;
            float lsum = 0.0f;

            for (int j = 0; j < Bc; ++j) {
                int pos = threadIdx.x * Bc + j;
                float val = (k_start + j < N) ? QK_T[pos] : -__FLT_MAX__;
                lmax = fmaxf(lmax, val);
            }

            for (int j = 0; j < Bc; ++j) {
                int pos = threadIdx.x * Bc + j;
                float val = (k_start + j < N) ? QK_T[pos] : -__FLT_MAX__;
                lsum += __expf(val - lmax);
            }

            local_max[threadIdx.x] = lmax;
            local_sum[threadIdx.x] = lsum;
        }
        __syncthreads();

        float tile_max = local_max[threadIdx.x];
        float tile_sum = local_sum[threadIdx.x];

        // ---- 3. Online merge of tile statistics into thread registers ----
        float curr_max = fmaxf(thread_max, tile_max);
        float exp_old_new = __expf(thread_max - curr_max);
        float curr_sum = thread_sum * exp_old_new + tile_sum;

        // Rescale previous accumulator
        thread_out = thread_out * exp_old_new * (thread_sum / curr_sum);

        // ---- 4. Add contribution from this tile:  (exp(QK - curr_max) / curr_sum) * V ----
        float weight = 0.0f;
        if (k_idx < N) {
            weight = __expf(qk - curr_max);
        }
        float v_val = (k_idx < N) ? inputV[k_idx * d + col] : 0.0f;
        thread_out += (weight / curr_sum) * v_val;

        // Update for next iteration
        thread_max = curr_max;
        thread_sum = curr_sum;

        __syncthreads();
    }

    // Final normalization
    if (thread_sum > 0.0f) {
        thread_out /= thread_sum;
    }

    output[row * d + col] = thread_out;
}