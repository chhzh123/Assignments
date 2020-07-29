#ifndef _INCL_CORE
#define _INCL_CORE

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define FULL_MASK 0xffffffff

#define CALLBACK1 cudaCallbackCPU
#define CALLBACK2 cudaCallbackGPU_baseline
#define CALLBACK3 cudaCallbackGPU_sharedmem
#define CALLBACK4 cudaCallbackGPU_reduction
#define CALLBACK5 cudaCallbackGPU_warp
#define CALLBACK6 cudaCallbackGPU_primitive
// #define CALLBACK7 ...
// #define CALLBACK8 ...
// #define CALLBACK9 ...
// #define CALLBACK10 ...

// The main function would invoke the "cudaCallback"s on each sample. Note that
// you don't have to (and shouldn't) free the space of searchPoints,
// referencePoints, and result by yourself since the main function have included
// the free statements already.
//
// To make the program work, you shouldn't modify the signature of\
// "cudaCallback"s.
extern void cudaCallbackCPU(int k, int m, int n, float *searchPoints,
                            float *referencePoints, int **results);
extern void cudaCallbackGPU_baseline(int k, int m, int n, float *searchPoints,
                                     float *referencePoints, int **results);
extern void cudaCallbackGPU_sharedmem(int k, int m, int n, float *searchPoints,
                                      float *referencePoints, int **results);
extern void cudaCallbackGPU_reduction(int k, int m, int n, float *searchPoints,
                                      float *referencePoints, int **results);
extern void cudaCallbackGPU_warp(int k, int m, int n, float *searchPoints,
                                      float *referencePoints, int **results);
extern void cudaCallbackGPU_primitive(int k, int m, int n, float *searchPoints,
                                      float *referencePoints, int **results);

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