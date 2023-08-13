#include "error.cuh"
#include <stdio.h>
#include <math.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int N = 100000000;
const int M = sizeof(real) * N;
const int NUM_REPEATS = 20;
const int BLOCK_SIZE = 128;
void timing(real *d_x, real *h_x);
__global__ void reduce_static(real *d_x, real *d_y);
__global__ void reduce_dynamic(real *d_x, real *d_y);

int main(void)
{
    // self-implemented reduction
    // Use -O3 option for highest level optimization
    // In float precision: nvcc -O3 -arch=sm_75 reduce1gpu_shared.cu -o reduce1gpu_shared.out
    // In double precision: nvcc -O3 -arch=sm_75 -DUSE_DP reduce1gpu_shared.cu -o reduce1gpu_shared.out
    // Run exe file: cuda-memcheck ./reduce1gpu_shared.out

    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
        h_x[n] = 1.23;

    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    timing(d_x, h_x);

    free(h_x);
    CHECK(cudaFree(d_x));

    return 0;
}

void timing(real *d_x, real *h_x)
{
    real sum = 0;
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        sum = 0;
        

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        
        const int block_size = BLOCK_SIZE;
        const int grid_size = ceil(real(N)/real(block_size));
        const int y_mem = grid_size * sizeof(real);

        real *d_y;
        CHECK(cudaMalloc(&d_y, y_mem));
        real *h_y = (real *) malloc(y_mem);
        
        // reduce_static<<<grid_size, block_size>>>(d_x, d_y);
        reduce_dynamic<<<grid_size, block_size, sizeof(real) * block_size>>>(d_x, d_y);
        
        CHECK(cudaMemcpy(h_y, d_y, y_mem, cudaMemcpyDeviceToHost));
        for(int i=0; i<grid_size; i++)
            sum += h_y[i];

        free(h_y);
        CHECK(cudaFree(d_y));

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

__global__ void reduce_static(real *d_x, real *d_y)
{
    __shared__ real s_y[BLOCK_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    s_y[tid] = bid * blockDim.x + tid < N ? d_x[bid * blockDim.x + tid] : 0;
    __syncthreads();

    for(int offset=blockDim.x>>1; offset > 0; offset>>=1)
    {
        if(tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid == 0)
        d_y[bid] = s_y[0];
      
}

__global__ void reduce_dynamic(real *d_x, real *d_y)
{
    extern __shared__ real s_y[];  // use real *s_y is illegal since address is not completely same with array name

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    s_y[tid] = bid * blockDim.x + tid < N ? d_x[bid * blockDim.x + tid] : 0;
    __syncthreads();

    for(int offset=blockDim.x>>1; offset > 0; offset>>=1)
    {
        if(tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid == 0)
        d_y[bid] = s_y[0];
      
}