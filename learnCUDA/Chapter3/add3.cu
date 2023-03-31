#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 4.56;
const double c = 5.79;
double __device__ add1device(double x, double y);
void __device__ add2device(double x, double y, double *z);
void __device__ add3device(double x, double y, double &z);
void __global__ add1(double *x, double *y, double *z, const int N);
void __global__ add2(double *x, double *y, double *z, const int N);
void __global__ add3(double *x, double *y, double *z, const int N);
void check(double *z, const int N);

int main(void)
{
    // Use CUDA Compiler: nvcc -arch=sm_75 add3.cu -o add
    // Run exe file: ./add
    // Codes run on GPU
    
    const int N = 100000001;
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
    cudaMalloc((void **) &d_x, M);  // Alloc memory on device
    cudaMalloc((void **) &d_y, M);  // force double** input &d_x to be void **
    cudaMalloc(&d_z, M);  // without void ** the cudaMalloc can finish transformation as well

    // Transfer data from host to device
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // grid_size=781250, it can exceeds max gridDim.x if you use CUDA 8.0 or lower version
    const int block_size = 128;
    const int grid_size = int((N + block_size - 1) / block_size);

    // params in <<<>>> have to be appoined
    add1<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    // Wrong transfer direction will cause wrong errors
    // cudaMemcpy(h_z, d_z, M, cudaMemcpyHostToDevice);
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    // notice the cuda data must be freed by cudaFree() 
    // and host data must be freed by free()
    // free(d_x);  // otherwise it would raise "Segmentation fault (core dumped)"
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(h_x);
    free(h_y);
    free(h_z);

    return 0;
}

// The number of inputs has to be fixed
// non-pointer inputs are visible for all threads
// pointer inputs have to point to device memory
// It can 
void __global__ add1(double *x, double *y, double *z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < N)
        z[n] = add1device(x[n], y[n]);
}

double __device__ add1device(double x, double y)
{
    return (x + y);
}

void __global__ add2(double *x, double *y, double *z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < N)
        add2device(x[n], y[n], &z[n]);
}

void __device__ add2device(double x, double y, double *z)
{
    *z = x + y;
}

void __global__ add3(double *x, double *y, double *z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < N)
        add3device(x[n], y[n], z[n]);
}

void __device__ add3device(double x, double y, double &z)
{
    z = x + y;
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

