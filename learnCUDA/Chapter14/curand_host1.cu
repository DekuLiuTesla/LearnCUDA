#include "error.cuh"
#include <curand.h>
#include <stdlib.h>
#include <stdio.h>

void output_results(int N, double *g_x);

int main()
{
    // Use -O3 option for highest level optimization
    // Compilation: nvcc -O3 -arch=sm_75 -lcurand curand_host1.cu -o curand_host1.out
    // Run exe file: cuda-memcheck ./curand_host1.out

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234);

    int N = 100000;
    double *g_x;
    cudaMalloc(&g_x, N * sizeof(double));
    curandGenerateUniformDouble(gen, g_x, N);

    double *x = (double *)malloc(N * sizeof(double));
    cudaMemcpy(x, g_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    output_results(N, x);
    
    cudaFree(g_x);
    curandDestroyGenerator(gen);
    free(x);

    return 0;
}

void output_results(int N, double *x)
{
    FILE *fid = fopen("x1.txt", "w");
    for (int i = 0; i < N; i++)
    {
        fprintf(fid, "%g\n", x[i]);
    }
    fclose(fid);
}