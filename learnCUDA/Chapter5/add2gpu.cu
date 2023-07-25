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
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void __global__ add(real *x, real *y, real *z, const int N);
void check(real *z, const int N);

int main(void)
{
    // Use -O3 option for highest level optimization
    // In float precision: nvcc -O3 -arch=sm_75 add2gpu.cu -o add2gpu.out
    // In real precision: nvcc -O3 -arch=sm_75 -DUSE_DP add2gpu.cu -o add2gpu.out
    // Run exe file: cuda-memcheck ./add2gpu.out

    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    real *h_z = (real*) malloc(M);

    for(int i=0;i<N;i++)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **) &d_x, M));  // Alloc memory on device
    CHECK(cudaMalloc((void **) &d_y, M));  // force real** input &d_x to be void **
    CHECK(cudaMalloc(&d_z, M));  // without void ** the cudaMalloc can finish transformation as well
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

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

        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

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

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    // notice the cuda data must be freed by cudaFree() 
    // and host data must be freed by free()
    // free(d_x);  // otherwise it would raise "Segmentation fault (core dumped)"
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    free(h_x);
    free(h_y);
    free(h_z);
    
    return 0;
}

// The number of inputs has to be fixed
// non-pointer inputs are visible for all threads
// pointer inputs have to point to device memory
void __global__ add(real *x, real *y, real *z, const int N)
{
    // mainly dominated by memory access
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    z[n] = x[n] + y[n];
}


void check(real *z, const int N)
{
    bool has_error = false;

    for(int i = 0; i < N; i++)
        if(abs(z[i] - c) > EPSILON)
        {
            has_error = true;
            break;
        }
    printf("%s\n", has_error ? "Has Errors" : "No Errors");
}
