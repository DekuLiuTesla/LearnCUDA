#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;  // can't directly use address or reference of it in kernel function

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void copy(const real *A, real *B, const int N);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
__global__ void transpose3(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    // Use CUDA Compiler: nvcc -arch=sm_75 matrix.cu -o matrix.out
    // Run exe file: ./matrix.out 10000
    if (argc != 2)
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);

    const int N2 = N * N;
    const int M = sizeof(real) * N2;
    real *h_A = (real *) malloc(M);
    real *h_B = (real *) malloc(M);
    for (int n = 0; n < N2; ++n)
    {
        h_A[n] = n;
    }
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ncopy:\n");
    timing(d_A, d_B, N, 0);
    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with shared coalesced read:\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with shared coalesced read and no bank conflicts:\n");
    timing(d_A, d_B, N, 3);

    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}

void timing(const real *d_A, real *d_B, const int N, const int task)
{
    const dim3 block_size(TILE_DIM, TILE_DIM);  // size of matrix patch

    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;  // to make sure N columns are covered
    const int grid_size_y = grid_size_x;  // to make sure N rows are covered
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task)
        {
            case 0:
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose3<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

__global__ void copy(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_in = ny * N + nx;
    const int idx_out = nx * N + ny;
    if (nx < N && ny < N)
    {
        B[idx_out] = A[idx_in];
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];

    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nx_in = bx + tx;
    const int ny_in = by + ty;
    const int idx_in = ny_in * N + nx_in;

    const int nx_out = by + tx;
    const int ny_out = bx + ty;
    const int idx_out = ny_out * N + nx_out;

    if (nx_in < N && ny_in < N)
    {
        S[ty][tx] = A[idx_in];
    }
    __syncthreads();
    
    if (nx_out < N && ny_out < N)
    {
        B[idx_out] = S[tx][ty];
    }       
}

__global__ void transpose3(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM + 1];

    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nx_in = bx + tx;
    const int ny_in = by + ty;
    const int idx_in = ny_in * N + nx_in;

    const int nx_out = by + tx;
    const int ny_out = bx + ty;
    const int idx_out = ny_out * N + nx_out;

    if (nx_in < N && ny_in < N)
    {
        S[ty][tx] = A[idx_in];
    }
    __syncthreads();
    
    if (nx_out < N && ny_out < N)
    {
        B[idx_out] = S[tx][ty];
    }      
}

void print_matrix(const int N, const real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}