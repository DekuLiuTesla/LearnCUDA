#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 4.56;
const double c = 5.79;
void add(double *x, double *y, double *z, const int N);
void check(double *z, const int N);

int main(void)
{
    // Use G++ Compiler: g++ add.cpp -o add
    // Run exe file: ./add
    
    const int N = 100000000;
    const int M = N * sizeof(double);

    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    for(int i=0;i<N;i++)
    {
        x[i] = a;
        y[i] = b;
    }

    add(x, y, z, N);
    check(z, N);

    free(x);
    free(y);
    free(z);

    return 0;
}

void add(double *x, double *y, double *z, const int N)
{
    for(int i = 0; i < N; i++)
        z[i] = x[i] + y[i];
}

void check(double *z, const int N)
{
    bool has_error = false;

    for(int i = 0; i < N; i++)
        if(abs(z[i] - c) > EPSILON)
        {
            has_error = true;
            break;
        }
    printf("%s\n", has_error ? "Has Errors" : "No Errors");
}

