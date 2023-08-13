#include "error.cuh"
#include <stdio.h>
#include <math.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
const int TILE_DIM = 32;
void timing(real *x, real *d_y, const int N, const int M);
__global__ void reduce(real *d_x, int N);

int main(void)
{
    // self-implemented reduction
    // Use -O3 option for highest level optimization
    // In float precision: nvcc -O3 -arch=sm_75 reduce1gpu.cu -o reduce1gpu.out
    // In double precision: nvcc -O3 -arch=sm_75 -DUSE_DP reduce1gpu.cu -o reduce1gpu.out
    // Run exe file: cuda-memcheck ./reduce1gpu.out

    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
        x[n] = 1.23;

    real *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMemcpy(d_x, x, M, cudaMemcpyHostToDevice));

    timing(d_x, d_y, N, M);

    free(x);
    CHECK(cudaFree(d_x));

    return 0;
}

void timing(real *d_x, real *d_y, const int N, const int M)
{
    real sum = 0;
    int n = N;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        sum = 0;
        n = N;

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        CHECK(cudaMemcpy(d_y, d_x, M, cudaMemcpyDeviceToDevice));
        while(n > 1)
        {
            int half_n = ceil(real(n)/2);
            const int block_size = TILE_DIM;
            const int grid_size = ceil(real(half_n)/real(block_size));
            
            reduce<<<grid_size, block_size>>>(d_y, n);
            n = half_n;
        }
        CHECK(cudaMemcpy(&sum, d_y, sizeof(real), cudaMemcpyDeviceToHost));
        printf("sum = %f.\n", sum);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}

__global__ void reduce(real *d_x, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int half_N = ceil(real(N)/2);
    if (tid < half_N)
    {
        if (tid == half_N - 1 && (N % 2 == 1))
            d_x[tid] = d_x[tid];
        else
            d_x[tid] += d_x[tid + half_N];
        // printf("d_x[%d] = %f, d_x[%d] = %f, N=%d, half_N=%d.\n", 
        //        tid, d_x[tid], tid + half_N, d_x[tid + half_N], N, half_N);
    }       
}