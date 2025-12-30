template <int Br, int Bc>
__global__ void attention_v2(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output)
{

    
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= N || col >= d) return;

    int steps = (N + Bc - 1) / Bc;

    __shared__ float QK_T[Br * Bc];
    __shared__ float SV[Br * Bc];
    __shared__ float block_max[Br * Bc];
    __shared__ float block_sum[Br * Bc];
    __shared__ float Vds[Bc * Bc];
    __shared__ float Qds[Br * Bc];
    __shared__ float Kds[Bc * Bc];
    
    float curr_max = -__FLT_MAX__;
    float prev_max = -__FLT_MAX__;
    float curr_sum = 1.0f;
    float prev_sum = 1.0f;

    float acc_o = 0.0f;
    for (int j = 0; j < steps; j++)
    {
        int start = threadIdx.y * Bc;
        int block_index = start + threadIdx.x;
        SV[block_index] = 0.0f;

        int k_row_index = threadIdx.x + j * Bc;
        float qk_dot = 0.0f;
        for (int ph = 0; ph < gridDim.x; ph++)
        {
            if (row < N && threadIdx.x + ph * Bc < d)
                Qds[block_index] = inputQ[row * d + threadIdx.x + ph * Bc];
            else
                Qds[block_index] = 0.0f;
            if (threadIdx.y < Bc)
                Kds[block_index] = 0.0f;
            if (threadIdx.y < Bc)
                if (k_row_index < N && threadIdx.y + ph * Bc < d)
                    Kds[block_index] = inputK[k_row_index * d + threadIdx.y + ph * Bc];

            __syncthreads();
            for (int index = 0; index < Bc; index++)
            {
                qk_dot = std::fma(Qds[start + index],
                                  Kds[index * Bc + threadIdx.x], qk_dot);
            }
            __syncthreads();
        }

        if (row < N && k_row_index < N)
        {
            block_max[block_index] = qk_dot;
            block_sum[block_index] = 1.0f;
            QK_T[block_index] = qk_dot;
        }
        else
        {
            block_max[block_index] = -__FLT_MAX__;
            block_sum[block_index] = 0.0f;
            QK_T[block_index] = 0.0f;
        }
        __syncthreads();

        for (int strip = Bc / 2; strip > 0; strip /= 2)
        {
            if (threadIdx.x < strip)
            {
                if (block_max[block_index] > block_max[block_index + strip])
                {
                    block_sum[block_index] = block_sum[block_index] + block_sum[block_index + strip] *
                            __expf(block_max[block_index + strip] - block_max[block_index]);
                }
                else
                {
                    block_sum[block_index] =
                        block_sum[block_index + strip] + block_sum[block_index] *
                            __expf(block_max[block_index] - block_max[block_index + strip]);
                    block_max[block_index] = block_max[block_index + strip];
                }
            }
            __syncthreads();
        }

        if (curr_max > block_max[start]) 
        {
            prev_sum = curr_sum;
            prev_max = curr_max;
            curr_sum = prev_sum + block_sum[start] *__expf(block_max[start] - curr_max);
        }
        else
        {
            prev_sum = curr_sum;
            prev_max = curr_max;
            curr_max = block_max[start];
            curr_sum = block_sum[start] + prev_sum * __expf(prev_max - curr_max);
        }

        if (threadIdx.y < Bc)
            if (threadIdx.y + j * Bc < N && col < d)
                Vds[threadIdx.x * Bc + threadIdx.y] = inputV[(threadIdx.y + j * Bc) * d + col];
            else
                Vds[threadIdx.x * Bc + threadIdx.y] = 0.0f;

        if (row < N && k_row_index < N)
            QK_T[block_index] = __expf(QK_T[block_index] - curr_max);
        else
            QK_T[block_index] = 0.0f;

        __syncthreads();

        for (int phc = 0; phc < Bc; phc++)
        {
            SV[block_index] = std::fma(
                QK_T[start + phc], Vds[threadIdx.x * Bc + phc],
                SV[block_index]);
        }
        acc_o = __expf(prev_max - curr_max) * acc_o * prev_sum + SV[block_index];
        acc_o = acc_o * __fdividef(1.0F, curr_sum);

        __syncthreads();
    }

    output[row * d + col] = acc_o;
}