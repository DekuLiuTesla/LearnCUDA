#include "error.cuh"
#include <stdio.h>
#include <stdint.h>

const int N = 30;

int main(void)
{
    // Use -O3 option for highest level optimization
    // Use global memory: nvcc -O3 -arch=sm_75 oversubscription1.cu -o oversubscription1.out
    // Use unified memory: nvcc -O3 -arch=sm_75 -DUNIFIED oversubscription1.cu -o oversubscription1.out
    // Run exe file: cuda-memcheck ./oversubscription1.out

    for (int n = 1; n <= N; ++n)
    {
        const size_t size = size_t(n) * 1024 * 1024 * 1024;
        uint64_t *x;
#ifdef UNIFIED
        CHECK(cudaMallocManaged(&x, size));
        CHECK(cudaFree(x));
        // will success even out of memory, since it only orders an address
        printf("Allocated %d GB unified memory without touch.\n", n);
#else
        CHECK(cudaMalloc(&x, size));
        CHECK(cudaFree(x));
        printf("Allocate %d GB device memory.\n", n);
#endif
    }
    return 0;
}

