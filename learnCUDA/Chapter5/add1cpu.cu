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
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main(void)
{
    // Use -O3 option for highest level optimization
    // In float precision: nvcc -O3 -arch=sm_75 add1cpu.cu -o add1cpu
    // In double precision: nvcc -O3 -arch=sm_75 -DUSE_DP add1cpu.cu -o add1cpu
    // Run exe file: cuda-memcheck ./add1cpu

    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real*) malloc(M);
    real *y = (real*) malloc(M);
    real *z = (real*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

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

        add(x, y, z, N);

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

    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const real *x, const real *y, real *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}
