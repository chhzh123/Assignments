/*
 * This code is part of the hw1 of multicore programming in SYSU
 * Copyright (c) 2020 Hongzheng Chen
 * Email: chenhzh37@mail2.sysu.edu.cn
 * 
 * This file is the kernel part of CUDA implementation
 * that calculates the central entropy of each point in a matrix.
 *
 * This program is an optimized implementation using shared memory.
 */

#include "core.h"

#define blockW 16
#define blockH 16
#define RADIUS 2
#define padW (blockW + 2 * RADIUS)
#define padH (blockH + 2 * RADIUS)

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
    const int x = blockIdx.x * blockW + threadIdx.x - RADIUS;
    const int y = blockIdx.y * blockH + threadIdx.y - RADIUS;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = y * width + x;
    __shared__ float smem[padH][padW];
    if (x >= 0 && x < width && y >= 0 && y < height) {
        smem[ty][tx] = input[idx];
    }
    __syncthreads();
    if (x >= 0 && x < width && y >= 0 && y < height &&
        tx >= RADIUS && tx < padW - RADIUS &&
        ty >= RADIUS && ty < padH - RADIUS) {
        int cnt[16] = {0};
        int valid = 0;
        for (int i = -2; i < 3; ++i)
            for (int j = -2; j < 3; ++j) {
                if (y + i >= 0 && y + i < height &&
                    x + j >= 0 && x + j < width) {
                    int in = smem[ty + i][tx + j];
                    cnt[in]++;
                    valid++;
                }
            }
        float sum = 0;
        for (int i = 0; i < 16; ++i) {
            int ni = cnt[i];
            if (ni != 0) {
                sum += ni * log_table[ni];
            }
        }
        output[idx] = -sum / valid + log_table[valid];
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
    const dim3 grid(divup(width, blockW), divup(height, blockH));
    const dim3 threadBlock(padW, padH);
    printf("grid: %d %d\n",divup(width, blockW),divup(height, blockH));
    printf("block size: %d %d\n",blockW,blockH);
    printf("pad block size (thread): %d %d\n",padW,padH);
    kernel<<< grid, threadBlock >>>(size, width, height, input_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *result = (float *)malloc(sizeof(float)*size);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));

    // Note that you don't have to free sample and *result by yourself
}