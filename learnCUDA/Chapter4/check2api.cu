#include "error.cuh"
#include <stdio.h>
#include <math.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 4.56;
const double c = 5.79;
void __global__ add(double *x, double *y, double *z, const int N);
void check(double *z, const int N);

int main(void)
{
    // Use CUDA Compiler: nvcc -arch=sm_75 check2api.cu -o check2api
    // Run exe file: ./check2api
    // Codes run on GPU
    
    const int N = 100000000;
    const int M = N * sizeof(double);

    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for(int i=0;i<N;i++)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **) &d_x, M));  // Alloc memory on device
    CHECK(cudaMalloc((void **) &d_y, M));  // force double** input &d_x to be void **
    CHECK(cudaMalloc(&d_z, M));  // without void ** the cudaMalloc can finish transformation as well

    // Transfer data from host to device
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    // now the block_size exceeds the max number of threads per block
    const int block_size = 1280;
    const int grid_size = int((N + block_size - 1) / block_size);

    // params in <<<>>> have to be appoined
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    // used to capture last error before cudaDeviceSynchronize()
    CHECK(cudaGetLastError());
    // synchronize to get accurate error location; same as using CUDA_LAUNCH_BLOCKING=1
    // now the host has to wait until all threads finish, so program will be slow
    CHECK(cudaDeviceSynchronize());  

    // Wrong transfer direction will cause wrong errors
    // CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyHostToDevice));
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
// It can 
void __global__ add(double *x, double *y, double *z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    z[n] = x[n] + y[n];
}


void check(double *z, const int N)
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

