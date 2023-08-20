#include "error.cuh"
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    // Use -O3 option for highest level optimization
    // Compilation: nvcc -O3 -arch=sm_75 -lcusolver cusolver.cu -o cusolver.out
    // Run exe file: cuda-memcheck ./cusolver.out

    int N = 2;
    int N2 = N * N;
    cuDoubleComplex *A_cpu = (cuDoubleComplex *)malloc(N2 * sizeof(cuDoubleComplex));
    A_cpu[0].x = 0;
    A_cpu[1].x = 0;
    A_cpu[2].x = 0;
    A_cpu[3].x = 0;
    A_cpu[0].y = 0; 
    A_cpu[1].y = 1;
    A_cpu[2].y = -1;
    A_cpu[3].y = 0;
    cuDoubleComplex *A_gpu;
    CHECK(cudaMalloc((void **)&A_gpu, N2 * sizeof(cuDoubleComplex)));
    CHECK(cudaMemcpy(A_gpu, A_cpu, N2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    double *W_cpu = (double *)malloc(N * sizeof(double));
    double *W_gpu;
    CHECK(cudaMalloc((void **)&W_gpu, N * sizeof(double)));

    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int lwork = 0;
    cusolverDnZheevd_bufferSize(handle, jobz, uplo, N, A_gpu, N, W_gpu, &lwork);
    cuDoubleComplex *work;
    CHECK(cudaMalloc((void **)&work, lwork * sizeof(cuDoubleComplex)));

    int *devInfo;
    CHECK(cudaMalloc((void **)&devInfo, sizeof(int)));
    cusolverDnZheevd(handle, jobz, uplo, N, A_gpu, N, W_gpu, work, lwork, devInfo);

    cudaMemcpy(W_cpu, W_gpu, N * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Eigenvalues:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", W_cpu[i]);
    }

    cusolverDnDestroy(handle);

    free(A_cpu);
    free(W_cpu);

    CHECK(cudaFree(A_gpu));
    CHECK(cudaFree(W_gpu));
    CHECK(cudaFree(work));
    CHECK(cudaFree(devInfo));

    return 0;
}
}