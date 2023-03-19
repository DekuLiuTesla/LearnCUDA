#include <stdio.h>
#include <iostream>

__global__ void hello_from_gpu()
{
    // std can't be used in device code
    // std::cout << "Hello World from the std::out\n";

    // blockIdx.x/y/z belongs to [0, gridDim.x/y/z - 1]
    // threadIdx.x/y/z belongs to [0, blockDim.x/y/z - 1]
    printf("Hello World from Block ID: %d, Thread ID: %d!\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
    // Use CUDA Compiler: nvcc hello3.cu -o hello
    // Run exe file: ./hello
    // Codes run on GPU

    // <<<grid_size, block_size>>> is required for running kernel
    // total_threads = grid_size * block_size, grid_size<=2^31-1, block_size<=1024, 
    // Computation Speed of different thread can be different
    hello_from_gpu<<<2, 4>>>();

    // Synchronize host and device
    cudaDeviceSynchronize();
    return 0;
}