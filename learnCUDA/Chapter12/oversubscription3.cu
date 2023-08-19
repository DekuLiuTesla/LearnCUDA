#include "error.cuh"
#include <stdio.h>
#include <stdint.h>

const int N = 96;

void cpu_touch(uint64_t *x, const size_t size)
{
    for(int idx = 0; idx < size; idx++)
        x[idx] = 0;
}

int main(void)
{
    // Use -O3 option for highest level optimization
    // Usage: nvcc -O3 -arch=sm_75 oversubscription3.cu -o oversubscription3.out
    // Run exe file: cuda-memcheck ./oversubscription3.out

    for (int n = 8; n <= N; n += 8)
    {
        const size_t memory_size = size_t(n) * 1024 * 1024 * 1024;
        const size_t data_size = memory_size / sizeof(uint64_t);
        uint64_t *x;
        CHECK(cudaMallocManaged(&x, memory_size));
        cpu_touch(x, data_size);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(x));
        printf("Allocated %d GB unified memory with CPU touch.\n", n);
    }
    return 0;
}

