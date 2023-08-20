#include "error.cuh"
#include <cublas_v2.h>
#include <stdio.h>

void print_matrix(int R, int C, double* A, const char* name);

int main()
{
    // Use -O3 option for highest level optimization
    // Compilation: nvcc -O3 -arch=sm_75 -lcublas cublas_gemm.cu -o cublas_gemm.out
    // Run exe file: cuda-memcheck ./cublas_gemm.out

    int N = 2;
    int M = 3;
    int K = 2;
    int NM = N * M;
    int MK = M * K;
    int NK = N * K;

    double* h_A = (double*)malloc(NM * sizeof(double));
    double* h_B = (double*)malloc(MK * sizeof(double));
    double* h_C = (double*)malloc(NK * sizeof(double));

    for (int i = 0; i < NM; i++)
        h_A[i] = i;
    print_matrix(N, M, h_A, "A");

    for (int i = 0; i < MK; i++)
        h_B[i] = i;
    print_matrix(M, K, h_B, "B");

    for (int i = 0; i < NK; i++)
        h_C[i] = 0;
    
    double *g_A, *g_B, *g_C;
    cudaMalloc(&g_A, NM * sizeof(double));
    cudaMalloc(&g_B, MK * sizeof(double));
    cudaMalloc(&g_C, NK * sizeof(double));

    cublasSetVector(NM, sizeof(double), h_A, 1, g_A, 1);
    cublasSetVector(MK, sizeof(double), h_B, 1, g_B, 1);
    cublasSetVector(NK, sizeof(double), h_C, 1, g_C, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M, &alpha, g_A, N, g_B, M, &beta, g_C, N);
    cublasDestroy(handle);

    cublasGetVector(NK, sizeof(double), g_C, 1, h_C, 1);
    print_matrix(N, K, h_C, "C");

    cudaFree(g_A);
    cudaFree(g_B);
    cudaFree(g_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

void print_matrix(int R, int C, double* A, const char* name)
{
    printf("%s = [\n", name);
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
            printf("%10.6f ", A[i + j * R]);
        printf("\n");
    }
    printf("]\n");
}