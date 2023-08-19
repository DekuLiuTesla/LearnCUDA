#include <stdio.h>

__device__ __managed__ int ret[1000];

__global__ void AplusB(int a, int b)
{
    ret[threadIdx.x] = a + b + threadIdx.x;
}

int main()
{
    // Use -O3 option for highest level optimization
    // nvcc -O3 -arch=sm_75 static.cu -o static.out
    // Run exe file: cuda-memcheck ./static.out

    AplusB<<<1, 1000>>>(10, 100);
    cudaDeviceSynchronize();
    for (int i = 0; i < 1000; i++)
    {
        printf("%d: A+B = %d\n", i, ret[i]);
    }
    return 0;
}