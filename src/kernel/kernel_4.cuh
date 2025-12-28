template <int Br, int Bc, int Rq>
__device__ void matmulRQK(const float *__restrict inputQ,
                          const float *__restrict inputK, float *shareQK,
                          float *shareVK, int N, int d, int width, int indQ,
                          int indK, float *regLeft, float *val)
{

    for (int ph = 0; ph < width; ph++)
    {
        if (threadIdx.y < Bc)
        {
            shareVK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        if (threadIdx.y < Bc)
        {
            if (indK < N && threadIdx.y + ph * Bc < d)
            {
                shareVK[threadIdx.y * Bc + threadIdx.x] =
                    inputK[indK * d + threadIdx.y + ph * Bc];
            }
        }

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + index_q < N && threadIdx.x + ph * Bc < d)
            {
                shareQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] =
                    inputQ[(indQ + index_q) * d + threadIdx.x + ph * Bc];
            }
            else
            {
                shareQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        for (int index = 0; index < Bc; index++)
        {
            for (int index_q = 0; index_q < Rq; index_q++)
            {
                regLeft[index_q] =
                    shareQK[(threadIdx.y * Rq + index_q) * Bc + index];
                val[index_q] =
                    std::fma(shareQK[(threadIdx.y * Rq + index_q) * Bc + index],
                             shareVK[index * Bc + threadIdx.x], val[index_q]);
            }
        }
        __syncthreads();
    }
}
template <int Br, int Bc, int Rq, int Rv>
__device__ void matmulSV(float *shareQK, const float *__restrict inputV,
                         float *shareVK, int N, int d, int j, int indQ,
                         int indK, int indV, float *val, float *newMax,
                         float *regLeft, float *regRight, float *sumSV)
{

    if (threadIdx.y < Bc)
    {
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            if (threadIdx.y + j * Bc < N && indV + index_v < d)
            {
                shareVK[threadIdx.y * Bc * Rv + threadIdx.x * Rv + index_v] =
                    inputV[(threadIdx.y + j * Bc) * d + indV + index_v];
            }
            else
            {
                shareVK[threadIdx.y * Bc * Rv + threadIdx.x * Rv + index_v] =
                    0.0f;
            }
        }
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {
        if (indQ + index_q < N && indK < N)
        {
            shareQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] =
                __expf(val[index_q] - newMax[index_q]);
        }
        else
        {

            shareQK[(threadIdx.y * Rq + index_q) * Bc + threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();
    for (int phc = 0; phc < Bc; phc++)
    {
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            regLeft[index_q] = shareQK[(threadIdx.y * Rq + index_q) * Bc + phc];
        }
        //--------
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            regRight[index_v] =
                shareVK[phc * Bc * Rv + threadIdx.x * Rv + index_v];
        }
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                sumSV[index_q * Rv + index_v] +=
                    regLeft[index_q] * regRight[index_v];
            }
        }
        // for (int index_q = 0; index_q < Rq; index_q++) {
        //     for (int index_v = 0; index_v < Rv; index_v++) {
        //         sumSV[index_q * Rv + index_v] +=
        //             shareQK[(threadIdx.y * Rq + index_q) * Bc + phc] *
        //             shareVK[phc * Bc * Rv + threadIdx.x * Rv + index_v];
        //     }
        // }
    }
}

template <int Br, int Bc, int Rq, int Rv>
__global__ void attention_v4(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output)
{

    __shared__ float shareQK[Rq * Br * Bc];
    __shared__ float shareVK[Bc * Bc * Rv];

    float sumSV[Rq * Rv];
    float newMax[Rq];
    float oldMax[Rq];
    float newSum[Rq];
    float regLeft[Rq];
    float regRight[Rv];
    float val[Rq];
    float regTmp[Rq];

    int indV = Rv * (threadIdx.x + blockIdx.x * blockDim.x);
    int indQ = Rq * (threadIdx.y + blockIdx.y * blockDim.y);

    for (int index_q = 0; index_q < Rq; index_q++)
    {
        newMax[index_q] = -__FLT_MAX__;
        oldMax[index_q] = -__FLT_MAX__;
        newSum[index_q] = 0.0f;
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            sumSV[index_q * Rv + index_v] = 0.0f;
        }
    }

    int Tc = (N + Bc - 1) / Bc;

    int width = (d + Bc - 1) / Bc;
    for (int j = 0; j < Tc; j++)
    {

        int indK = threadIdx.x + j * Bc;
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            val[index_q] = 0.0f;
        }

        matmulRQK<Br, Bc, Rq>(inputQ, inputK, shareQK, shareVK, N, d, width,
                              indQ, indK, regLeft, val);
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + index_q < N && indK < N)
            {

                regTmp[index_q] = val[index_q];
            }
            else
            {

                regTmp[index_q] = -__FLT_MAX__;
            }
        }
        __syncthreads();
        // softmax reduce
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            regTmp[index_q] = WarpAllReduce<MaxOp, float, Bc>(regTmp[index_q]);
            if (threadIdx.x == 0)
            {
                shareQK[threadIdx.y * Rq + index_q] = regTmp[index_q];
            }
        }
        __syncthreads();
        //--------------------
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (indQ + index_q < N && indK < N)
            {
                regTmp[index_q] =
                    __expf(val[index_q] - shareQK[threadIdx.y * Rq + index_q]);
            }
            else
            {

                regTmp[index_q] = 0.0f;
            }
        }
        __syncthreads();
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            regTmp[index_q] = WarpAllReduce<SumOp, float, Bc>(regTmp[index_q]);
            if (threadIdx.x == 0)
            {
                shareQK[threadIdx.y * Rq + index_q + Rq * Br] = regTmp[index_q];
            }
        }
        __syncthreads();
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            if (newMax[index_q] > shareQK[threadIdx.y * Rq + index_q])
            {
                newSum[index_q] =
                    std::fma(shareQK[threadIdx.y * Rq + index_q + Rq * Br],
                             __expf(shareQK[threadIdx.y * Rq + index_q] -
                                    newMax[index_q]),
                             newSum[index_q]);
            }
            else
            {
                newSum[index_q] =
                    std::fma(newSum[index_q],
                             __expf(newMax[index_q] -
                                    shareQK[threadIdx.y * Rq + index_q]),
                             shareQK[threadIdx.y * Rq + index_q + Rq * Br]);

                newMax[index_q] = shareQK[threadIdx.y * Rq + index_q];
            }
        }
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                sumSV[index_q * Rv + index_v] *=
                    __expf(oldMax[index_q] - newMax[index_q]);
            }
        }
        matmulSV<Br, Bc, Rq, Rv>(shareQK, inputV, shareVK, N, d, j, indQ, indK,
                                 indV, val, newMax, regLeft, regRight, sumSV);

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            oldMax[index_q] = newMax[index_q];
        }

        //__syncthreads();
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {
        float inv = __fdividef(1.0F, newSum[index_q]);
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            if (indQ + index_q < N && indV + index_v < d)
            {
                output[(indQ + index_q) * d + indV + index_v] =
                    sumSV[index_q * Rv + index_v] * inv;
            }
        }
    }
}