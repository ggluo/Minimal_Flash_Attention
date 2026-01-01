#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <utils.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void save_matrix_to_file(const float* matrix, int size, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    // Save with high precision
    for (int i = 0; i < size; i++) {
        fprintf(file, "%.8f\n", matrix[i]);
    }
    
    fclose(file);
    printf("Saved output to %s\n", filename);
}

void load_matrix_from_file(float* matrix, int size, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &matrix[i]) != 1) {
            printf("Error: Failed to read element %d from %s\n", i, filename);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    
    fclose(file);
    printf("Loaded input from %s\n", filename);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Please select a kernel (range 0 - 7).\n");
        exit(EXIT_FAILURE);
    }

    // cuda kernel num
    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 7)
    {
        printf("Please enter a valid kernel number (0-7).\n");
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
    
    // Load matrices from Python-generated files
    load_matrix_from_file(cpu_Q, size, "python_input_Q.txt");
    load_matrix_from_file(cpu_K, size, "python_input_K.txt");
    load_matrix_from_file(cpu_V, size, "python_input_V.txt");
    
    // Initialize output with zeros
    for (int i = 0; i < size; i++) {
        cpu_output[i] = 0.0f;
    }

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

    // Save output to file
    char filename[256];
    snprintf(filename, sizeof(filename), "kernel%d_output.txt", kernel_num);
    save_matrix_to_file(cpu_output, N * d, filename);

    cudaFree(gpu_Q);
    cudaFree(gpu_K);
    cudaFree(gpu_V);
    cudaFree(gpu_output);
    
    printf("Average elapsed time: (%f) ms.\n",
           elapsed_time / repeat);

    free(cpu_Q);
    free(cpu_K);
    free(cpu_V);
    free(cpu_output);

    return 0;
}
