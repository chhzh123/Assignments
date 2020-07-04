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
}

static __constant__ float log_table[26] = {0, // do not use index 0!
                                           0.0,
                                           0.6931471805599453,
                                           1.0986122886681098,
                                           1.3862943611198906,
                                           1.6094379124341003,
                                           1.791759469228055,
                                           1.9459101490553132,
                                           2.0794415416798357,
                                           2.1972245773362196,
                                           2.302585092994046,
                                           2.3978952727983707,
                                           2.4849066497880004,
                                           2.5649493574615367,
                                           2.6390573296152584,
                                           2.70805020110221,
                                           2.772588722239781,
                                           2.833213344056216,
                                           2.8903717578961645,
                                           2.9444389791664403,
                                           2.995732273553991,
                                           3.044522437723423,
                                           3.091042453358316,
                                           3.1354942159291497,
                                           3.1780538303479458,
                                           3.2188758248682006};

#endif