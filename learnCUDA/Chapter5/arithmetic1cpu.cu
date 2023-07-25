#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real x0 = 100.0;
void arithmetic(real *x, const real x0, const int N);

int main(void)
{
    // Use -O3 option for highest level optimization
    // In float precision: nvcc -O3 -arch=sm_75 arithmetic1cpu.cu -o arithmetic1cpu.out
    // In double precision: nvcc -O3 -arch=sm_75 -DUSE_DP arithmetic1cpu.cu -o arithmetic1cpu.out
    // Run exe file: cuda-memcheck ./arithmetic1cpu.out

    const int N = 10000;
    const int M = sizeof(real) * N;
    real *x = (real*) malloc(M);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        // make sure cudaEventRecord run on GPU in WDDM mode
        cudaEventQuery(start);

        arithmetic(x, x0, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            // abort first iter since cpu and gpu warm up cost more time
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;  // mean
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);  // std
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    free(x);

    return 0;
}

void arithmetic(real *x, const real x0, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        real x_tmp = x[n];
        while(sqrt(x_tmp) < x0)
            ++x_tmp;
        x[n] = x_tmp;
    }
}
