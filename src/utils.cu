#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

float get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) {
    return 1.0e-6 * (end - beg);
}

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void randomize_matrix(float *mat, int N) {
    // NOTICE: 使用gettimeofdays替代srand((unsigned)time(NULL));time精度过低，产生相同随机数
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N) {
    int i;
    printf("[");
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

template <int Br, int Bc>
void test_cublas_attention(cublasHandle_t handle,
                        const float *__restrict inputQ,
                        const float *__restrict inputK,
                        const float *__restrict inputV, int N, int d, 
                        float *__restrict output) {
    printf("test_cublas_attention\n");
}

template <int Br, int Bc>
void test_attention_v1(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, 
                       float *__restrict output) {
    int num_block_x = (N + Br - 1) / Br;
    int num_block_y = (d + Bc - 1) / Bc;
    dim3 block_dim(Br, Bc, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    int share_mem = 4 * Br * Bc * sizeof(float);
    attention_v1<<<grid_dim, block_dim, share_mem>>>(inputQ, inputK, inputV, N, d, Br, Bc, output);
}

template <int Br, int Bc>
void test_attention_v2(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, 
                       float *__restrict output) {
    int num_block_x = (d + Bc - 1) / Bc;
    int num_block_y = (N + Br - 1) / Br;
    dim3 block_dim(Bc, Br, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    attention_v2<Br, Bc><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}

template <int Br, int Bc>
void test_attention_v3(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, 
                       float *__restrict output) {
    const int Rq = 2;
    const int Rv = 4;
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);
    attention_v3<Br, Bc, Rq, Rv><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}

template <int Br, int Bc>
void test_attention_v4(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, 
float *__restrict output) {
    const int Rq = 3;
    const int Rv = 4;
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);
    attention_v4<Br, Bc, Rq, Rv><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}

template <int Br, int Bc>
void test_attention_v5(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, 
float *__restrict output) {
    const int Rq = 3;
    const int Rv = 4;
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);

    attention_v5<Br, Bc, Rq, Rv><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}

template <int Br, int Bc>
void test_attention_v6(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, 
float *__restrict output) {
    const int Rq = 8;
    const int Rv = 8; // 必须是4的倍数
    const int Bk = 8; // 必须是4的倍数
    const int Bd = 8;
    const int numQ = Rq * Br;
    const int numK = Bk * Bc;
    const int numV = Rv * Bc;

    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);

    attention_v6<16, 16, 8, 8, 8, 8, 8*16, 8*16, 8*16><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}

template <int Br, int Bc>
void test_attention_v7(const float *__restrict inputQ,
                       const float *__restrict inputK,
                       const float *__restrict inputV, int N, int d, float *__restrict output) {

    const int Rq = 8;
    const int Rv = 8; // 必须是4的倍数
    const int Bk = 8; // 必须是4的倍数
    const int Bd = 8;
    const int numQ = Rq * Br;
    const int numK = Bk * Bc;
    const int numV = Rv * Bc;
    
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);

    attention_v7<Br, Bc, Rq, Rv, Bd, Bk, numQ, numK, numV><<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}

void test_kernel(const float *__restrict inputQ,
                const float *__restrict inputK,
                const float *__restrict inputV,
                float *__restrict output,int N, int d,
                int kernel_num, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            //test_cublas(handle, M, N, K, alpha, A, B, beta, C);
            printf("skip cublas\n");
            break;
        case 1:
            test_attention_v1<32, 32>(inputQ, inputK, inputV, N, d, output);
            break;
        case 2:
            test_attention_v2<32, 32>(inputQ, inputK, inputV, N, d, output);
            break;
        case 3:
            test_attention_v3<32, 32>(inputQ, inputK, inputV, N, d, output);
            break;
        case 4:
            test_attention_v4<32, 32>(inputQ, inputK, inputV, N, d, output);
            break;
        case 5:
            test_attention_v5<32, 32>(inputQ, inputK, inputV, N, d, output);
            break;
        case 6:
            test_attention_v6<16, 16>(inputQ, inputK, inputV, N, d, output);
            break;
        case 7:
            test_attention_v7<16, 16>(inputQ, inputK, inputV, N, d, output);
            break;
        default:
            break;
    }
}