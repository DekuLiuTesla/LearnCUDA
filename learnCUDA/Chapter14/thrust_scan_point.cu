#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <stdio.h>

int main(void)
{
    // Use -O3 option for highest level optimization
    // Compilation: nvcc -O3 -arch=sm_75 thrust_scan_point.cu -o thrust_scan_point.out
    // Run exe file: cuda-memcheck ./thrust_scan_point.out

    int N = 10;
    // allocate two device_vectors with 10 elements
    int *x, *y;
    cudaMallocManaged(&x, N*sizeof(int));
    cudaMallocManaged(&y, N*sizeof(int));

    // initialize x to 1,2,3,4 ....
    thrust::sequence(thrust::device, x, x + N, 1);

    // inclusive scan x and store result in y
    thrust::inclusive_scan(thrust::device, x, x + N, y);

    // print y
    for(int i = 0; i < N; i++)
        std::cout << "y[" << i << "] = " << y[i] << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}