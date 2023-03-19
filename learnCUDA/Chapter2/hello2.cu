#include <stdio.h>
#include <iostream>

__global__ void hello_from_gpu()
{
    // std can't be used in device code
    // std::cout << "Hello World from the std::out\n";
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    // Use CUDA Compiler: nvcc hello2.cu -o hello
    // Run exe file: ./hello
    // Codes run on GPU

    // <<<grid_size, block_size>>> is required for running kernel
    // total_threads = grid_size * block_size
    hello_from_gpu<<<2, 3>>>();

    // Synchronize host and device
    cudaDeviceSynchronize();
    return 0;
}