#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <stdio.h>

int main(void)
{
    // Use -O3 option for highest level optimization
    // Compilation: nvcc -O3 -arch=sm_75 thrust_scan_vector.cu -o thrust_scan_vector.out
    // Run exe file: cuda-memcheck ./thrust_scan_vector.out

    int N = 10;
    // allocate two device_vectors with 10 elements
    thrust::device_vector<int> x(N, 0);
    thrust::device_vector<int> y(N, 0);

    // initialize x to 1,2,3,4 ....
    thrust::sequence(x.begin(), x.end(), 1);

    // inclusive scan x and store result in y
    thrust::inclusive_scan(x.begin(), x.end(), y.begin());

    // print y
    for(int i = 0; i < y.size(); i++)
        std::cout << "y[" << i << "] = " << y[i] << std::endl;

    return 0;
}