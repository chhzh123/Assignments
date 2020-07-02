#include "core.h"

__global__ void kernel(int size, int width, int height, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
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
            float ni = cnt[i];
            if (ni != 0) {
                sum += ni * logf(ni);
            }
        }
        output[idx] = -sum / valid + logf(valid);
    }
}

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