#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <utils.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Please select a kernel (range 0 - 7).\n");
        exit(EXIT_FAILURE);
    }

    // cuda kernel num
    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 8)
    {
        printf("Please enter a valid kernel number (0-8).\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Select kernel %d.\n", kernel_num);
    };

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    int N = 1024;
    int d = 1024;

    int size = N * d;

    float *cpu_Q=NULL, *cpu_K=NULL, *cpu_V=NULL, *cpu_output=NULL;
    cpu_Q = (float *)malloc(size * sizeof(float));
    cpu_K = (float *)malloc(size * sizeof(float));
    cpu_V = (float *)malloc(size * sizeof(float));
    cpu_output = (float *)malloc(size * sizeof(float));
    
    randomize_matrix(cpu_Q, size);
    randomize_matrix(cpu_K, size);
    randomize_matrix(cpu_V, size);
    randomize_matrix(cpu_output, size);

    float *gpu_Q=NULL, *gpu_K=NULL, *gpu_V=NULL, *gpu_output=NULL;

    cudaCheck(cudaMalloc((void **)&gpu_Q, size * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&gpu_K, size * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&gpu_V, size * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&gpu_output, size * sizeof(float)));

    cudaCheck(cudaMemcpy(gpu_Q, cpu_Q, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu_K, cpu_K, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu_V, cpu_V, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu_output, cpu_output, size * sizeof(float), cudaMemcpyHostToDevice));

    int repeat = 10;
    
    test_kernel(gpu_Q, gpu_K, gpu_V, gpu_output, N, d, 0, true);
    cudaEventRecord(beg);
    for (int i = 0; i < repeat; i++)
    {
        test_kernel(gpu_Q, gpu_K, gpu_V, gpu_output, N, d, kernel_num, false);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    cudaMemcpy(cpu_output, gpu_output, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_Q);
    cudaFree(gpu_K);
    cudaFree(gpu_V);
    cudaFree(gpu_output);
    
    printf("Average elasped time: (%f) millisecond.\n",
           elapsed_time / repeat);

    free(cpu_Q);
    free(cpu_K);
    free(cpu_V);
    free(cpu_output);

    return 0;
}
