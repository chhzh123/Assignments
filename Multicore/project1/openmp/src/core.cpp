/*
 * This code is part of the hw1 of multicore programming in SYSU
 * Copyright (c) 2020 Hongzheng Chen
 * Email: chenhzh37@mail2.sysu.edu.cn
 * 
 * This file is the kernel part of OpenMP implementation
 *   that calculates the central entropy of each point in a matrix
 */

#include "core.h"

/*!
 * Core execution part of OpenMP
 *   that calculates the central entropy of each point.
 * \param width The width of the input matrix.
 * \param height The height of the input matrix.
 * \param input The input matrix.
 * \param output The output matrix.
 * \return void. Results will be put in output.
 */
void kernel(int width, int height, float *input, float *output) {
    #pragma omp parallel for
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = h * width + w;
            int cnt[16] = {0};
            int valid = 0;
            int x = idx % width;
            int y = idx / width;
            for (int i = -2; i < 3; ++i)
                for (int j = -2; j < 3; ++j) {
                    if (y + i >= 0 && y + i < height &&
                        x + j >= 0 && x + j < width) {
                        int in = input[idx + i * width + j];
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
}

/*!
 * Wrapper of the OpenMP kernel
 *   used to be called in the main function
 * \param width The width of the input matrix.
 * \param height The height of the input matrix.
 * \param sample The input matrix.
 * \param result The output matrix.
 * \return void. Results will be put in result.
 */
void openmpCallback(int width, int height, float *sample, float **result) {
    int size = width * height;

    // Allocate device memory and copy data from host to device
    float* input_d = (float*) malloc(sizeof(float) * size);
    float* output_d = (float*) malloc(sizeof(float) * size);
    memcpy(input_d, sample, sizeof(float) * size);

    // Invoke the device function
    kernel(width, height, input_d, output_d);

    // Copy back the results and de-allocate the device memory
    *result = (float*) malloc(sizeof(float) * size);
    memcpy(*result, output_d, sizeof(float) * size);
    free(input_d);
    free(output_d);
}