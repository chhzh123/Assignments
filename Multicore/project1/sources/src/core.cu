/*
 * This code is part of the hw1 of multicore programming in SYSU
 * Copyright (c) 2020 Hongzheng Chen
 * Email: chenhzh37@mail2.sysu.edu.cn
 * 
 * This file is the kernel part of CUDA implementation
 * that calculates the central entropy of each point in a matrix.
 *
 * This code is a baseline. Only constant lookup table is added,
 * and no other optimizations are enabled.
 */

#include "core.h"
// #define LOOKUP

/*!
 * Core execution part of CUDA
 *   that calculates the central entropy of each point.
 * \param size The size of the input matrix.
 * \param width The width of the input matrix.
 * \param height The height of the input matrix.
 * \param input The input matrix.
 * \param output The output matrix.
 * \return void. Results will be put in output.
 */
__global__ void kernel(int size, int width, int height, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int cnt[16] = {0};
        int valid = 0;
        int x = idx % width;
        int y = idx / width;
        // each thread first counts the histogram of idx
        for (int i = -2; i < 3; ++i)
            for (int j = -2; j < 3; ++j) {
                if (y + i >= 0 && y + i < height &&
                    x + j >= 0 && x + j < width) {
                    int in = input[idx + i * width + j];
                    cnt[in]++;
                    valid++;
                }
            }
        // calculate entropy
        float sum = 0;
        for (int i = 0; i < 16; ++i) {
            int ni = cnt[i];
            if (ni != 0) {
                #ifdef LOOKUP
                sum += ni * log_table[ni];
                #else
                sum += ni * logf(ni);
                #endif
            }
        }
        #ifdef LOOKUP
        output[idx] = -sum / valid + log_table[valid];
        #else
        output[idx] = -sum / valid + logf(valid);
        #endif
    }
}

/*!
 * Wrapper of the CUDA kernel
 *   used to be called in the main function
 * \param width The width of the input matrix.
 * \param height The height of the input matrix.
 * \param sample The input matrix.
 * \param result The output matrix.
 * \return void. Results will be put in result.
 */
void cudaCallback(int width, int height, float *sample, float **result) {
    int size = width * height;
    float *input_d, *output_d;

    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&input_d, sizeof(float)*size));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float)*size));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float)*size, cudaMemcpyHostToDevice));

    // Invoke the device function
    kernel<<< divup(size, 1024), 1024 >>>(size, width, height, input_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *result = (float *)malloc(sizeof(float)*size);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));

    // Note that you don't have to free sample and *result by yourself
}