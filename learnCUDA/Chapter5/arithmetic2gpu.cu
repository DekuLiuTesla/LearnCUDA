#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real x0 = 100.0;
void __global__ arithmetic(real *d_x, const real x0, const int N);

int main(int argc, char **argv)
{
    // Use -O3 option for highest level optimization
    // In float precision: nvcc -O3 -arch=sm_75 arithmetic2gpu.cu -o arithmetic2gpu.out
    // In double precision: nvcc -O3 -arch=sm_75 -DUSE_DP arithmetic2gpu.cu -o arithmetic2gpu.out
    // Run exe file: cuda-memcheck ./arithmetic2gpu.out 1000000

    if (argc != 2) 
    {
        // argc is the number of arguments
        // argv keeps the arguments as char arrays, argv[0] is the name of the program
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    // Use int atoi(const char *str) change string to int
    const int N = atoi(argv[1]);
    const int M = sizeof(real) * N;
    real *h_x = (real*) malloc(M);

    real *d_x;
    CHECK(cudaMalloc((void **) &d_x, M));  // Alloc memory on device
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    // grid_size=781250, it can exceeds max gridDim.x if you use CUDA 8.0 or lower version
    const int block_size = 128;
    const int grid_size = int((N + block_size - 1) / block_size);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        // make sure cudaEventRecord run on GPU in WDDM mode
        cudaEventQuery(start);

        arithmetic<<<grid_size, block_size>>>(d_x, x0, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            // abort first iter since cpu and gpu warm up cost more time
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;  // mean
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);  // std
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    // notice the cuda data must be freed by cudaFree() 
    // and host data must be freed by free()
    // free(d_x);  // otherwise it would raise "Segmentation fault (core dumped)"
    CHECK(cudaFree(d_x));
    free(h_x);
    
    return 0;
}

// The number of inputs has to be fixed
// non-pointer inputs are visible for all threads
// pointer inputs have to point to device memory
void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    // mainly dominated by memory access
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        real x_tmp = d_x[n];
        while(sqrt(x_tmp) < x0)
            ++x_tmp;
        d_x[n] = x_tmp;
    }
}
