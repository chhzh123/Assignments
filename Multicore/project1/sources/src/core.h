#ifndef _INCL_CORE
#define _INCL_CORE

#include <stdio.h>

// kernel is an element-wise kernel function for demonstrating purpose only.
extern __global__ void kernel(int size, int width, int height, float *input, float *output);

/*
 * In general, you don't need to modify the lines below to finish hw1.
 */

// The main function would invoke the cudaCallback on each sample. Note that you
// don't have to (and shouldn't) free the space of sample and result by yourself
// since the main function have included the free statements already.
//
// To make the program work, you shouldn't modify the signature of cudaCallback.
extern void cudaCallback(int width, int height, float *sample, float **result);

// divup calculates n / m and would round it up if the remainder is non-zero.
extern int divup(int n, int m);

// CHECK macro from Grossman and McKercher, "Professional CUDA C Programming"
#define CHECK(call)                                         \
{                                                           \
    const cudaError_t error = call;                         \
    if (error != cudaSuccess) {                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);       \
        printf("code:%d, reason: %s \n",                    \
                error, cudaGetErrorString(error));          \
        exit(1);                                            \
    }                                                       \
}                                                           \

#endif