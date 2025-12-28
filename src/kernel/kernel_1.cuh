#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

__global__ void attention_v1(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d, int Br, int Bc,
                                 float *__restrict output)
{
    // 一个线程块处理Q的Br行，V的Bc列，以及全部的K,blockDim.x=Br,blockDim.y=Bc
    int Tc = (N + Bc - 1) / Bc;                       // 遍历矩阵inputK的N行需要的循环次数
    extern __shared__ float sram[];                   // 必须要有extern
    float *block_sum = sram;                          // 形状为[Br,Bc]，为后面softmax计算sum做准备
    float *block_max = sram + Br * Bc;                // 形状为[Br,Bc]，为后面softmax计算max做准备
    float *sumQK = sram + Br * Bc * 2;                // 形状为[Br,Bc]，存储的是QK.T的结果
    float *sumSV = sram + Br * Bc * 3;                // 形状为[Br,Bc]，存储的是softmax(QK.T)V的结果
    int indQ = threadIdx.x + blockIdx.x * blockDim.x; // 对应的是当前block需要处理的Q的行索引
    int indV = threadIdx.y + blockIdx.y * blockDim.y; // 对应的是当前block需要处理的V的列索引
    float newMax;                                     // newMax就是算法里面的m_{i,j}
    float oldMax;
    float newSum; // newSum就是l_{i,j}

    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 0.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++)
    {
        sumSV[threadIdx.x * Bc + threadIdx.y] = 0.0f; // 每次循环需要重新初始化为0
        int indK = threadIdx.y + j * Bc;              // 通过j循环来遍历K的行索引
        float sum_qk = 0.0f;
        for (int index = 0; index < d; index++)
        {
            sum_qk += inputQ[indQ * d + index] * inputK[indK * d + index];
        }
        if (indQ < N && indK < N)
        {

            block_max[threadIdx.x * Bc + threadIdx.y] = sum_qk; // 后面针对threadIdx.y做规约会修改元素内容
            sumQK[threadIdx.x * Bc + threadIdx.y] = sum_qk;     // 存储QK的结果，循环内部不做修改
            block_sum[threadIdx.x * Bc + threadIdx.y] = 1.0f;
        }
        else
        {
            sumQK[threadIdx.x * Bc + threadIdx.y] = 0.0f;
            block_max[threadIdx.x * Bc + threadIdx.y] = -__FLT_MAX__;
            block_sum[threadIdx.x * Bc + threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int strip = Bc / 2; strip > 0; strip /= 2) // 这部分规约可以理解为二维block的softmax规约，一边算max，一边算sum
        {
            if (threadIdx.y < strip)
            {
                if (block_max[threadIdx.x * Bc + threadIdx.y] >
                    block_max[threadIdx.x * Bc + threadIdx.y + strip])
                {
                    block_sum[threadIdx.x * Bc + threadIdx.y] =
                        block_sum[threadIdx.x * Bc + threadIdx.y] +
                        block_sum[threadIdx.x * Bc + threadIdx.y + strip] *
                            __expf(block_max[threadIdx.x * Bc + threadIdx.y + strip] -
                                   block_max[threadIdx.x * Bc + threadIdx.y]);
                }
                else
                {
                    block_sum[threadIdx.x * Bc + threadIdx.y] =
                        block_sum[threadIdx.x * Bc + threadIdx.y + strip] +
                        block_sum[threadIdx.x * Bc + threadIdx.y] *
                            __expf(block_max[threadIdx.x * Bc + threadIdx.y] -
                                   block_max[threadIdx.x * Bc + threadIdx.y + strip]);
                    block_max[threadIdx.x * Bc + threadIdx.y] =
                        block_max[threadIdx.x * Bc + threadIdx.y + strip];
                }
            }
            __syncthreads();
        } // 规约结果存储在threadIdx.y=0的位置
        if (newMax > block_max[threadIdx.x * Bc]) // threadIdx.y=0存储的是对应分块矩阵的局部max
        {                                         // 为了获得全局max，需要不断更新newMax和threadIdx.y=0的比较结果
            newSum = newSum + block_sum[threadIdx.x * Bc] *
                                  __expf(block_max[threadIdx.x * Bc] - newMax);
        }
        else
        {
            newSum = block_sum[threadIdx.x * Bc] +
                     newSum * __expf(newMax - block_max[threadIdx.x * Bc]);
            newMax = block_max[threadIdx.x * Bc];
        }

        __syncthreads();
        for (int phc = 0; phc < Bc; phc++) // 这里开始做最后和V的matmul
        {
            if (phc + j * Bc < N) // 注意控制范围
            {
                sumSV[threadIdx.x * Bc + threadIdx.y] += __expf(sumQK[threadIdx.x * Bc + phc] - newMax) * inputV[(phc + j * Bc) * d + indV];
            }
        }
        out = __expf(oldMax - newMax) * out + sumSV[threadIdx.x * Bc + threadIdx.y];
        oldMax = newMax;
        __syncthreads();
    }
    if (indQ < N && indV < d)
    {
        output[indQ * d + indV] = out * __fdividef(1.0F, newSum);
    }
}