#pragma once  // avoid repeat including
#include <stdio.h>

// do{...}while(0) is safer for macro definition and generality, but it has no return
// Comment can't be put in the macro definition
// \ is required if you can't put all macro definition in one line
#define CHECK(call) \
do{ \
    const cudaError_t error_code = call; \
    if(error_code != cudaSuccess) \
    { \
        printf("CUDA Error: \n"); \
        printf("    File:    %s\n", __FILE__); \
        printf("    Line:    %d\n", __LINE__); \
        printf("    Error Code:    %d\n", error_code); \
        printf("    Error Text:    %s\n", cudaGetErrorString(error_code)); \
        exit(1); \
    } \
}while(0)
