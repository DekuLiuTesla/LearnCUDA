#include <stdio.h>
#include <iostream>

__global__ void hello_from_gpu()
{
    // std can't be used in device code
    // std::cout << "Hello World from the std::out\n";

    // blockIdx.x/y/z belongs to [0, gridDim.x/y/z - 1]
    // threadIdx.x/y/z belongs to [0, blockDim.x/y/z - 1]
    
    // flattened nearest 32 threads consists of a thread wrap
    const int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    printf("Hello World from block-%d, thread-%d and thread-(%d, %d)!\n", bid, tid, threadIdx.x, threadIdx.y);
}

int main(void)
{
    // Use CUDA Compiler: nvcc hello4.cu -o hello
    // Run exe file: ./hello
    // Codes run on GPU

    // <<<grid_size, block_size>>> is required for running kernel
    // total_threads = grid_size * block_size
    // gridDim.x <= 2^31-1, gridDim.y/z <= 65536
    // blockDim.x/y <= 1024, blockDim.z <= 64
    // Computation Speed of different thread can be different
    const dim3 block_size(2, 4);
    hello_from_gpu<<<2, block_size>>>();

    // Synchronize host and device
    cudaDeviceSynchronize();
    return 0;
}