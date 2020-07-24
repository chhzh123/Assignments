/*
 * This code is part of the hw2 of multicore programming in SYSU
 * Copyright (c) 2020 Hongzheng Chen
 * Email: chenhzh37@mail2.sysu.edu.cn
 * 
 * This file is the kernel part of CUDA implementation
 * of calculating nearest neighbor in high dimension.
 *
 * This program is a baseline implementation.
 */

#include "core.h"
#define MAX_Q 1024

/*!
 * Naive CPU implementation
 *   used to test the correctness of the results
 * \param k The dimension size of the points
 * \param m The nubmer of search points
 * \param n The number of reference points
 * \param searchPoints
 * \param referencePoints
 * \param results
 * \return void. Results will be put in result.
 */
extern void cudaCallbackCPU(int k, int m, int n, float *searchPoints,
                            float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);
    int minIndex;
    float minSquareSum, diff, squareSum;

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k*mInd+kInd] - referencePoints[k*nInd+kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        tmp[mInd] = minIndex;
    }

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}

/*!
 * Core execution part of CUDA
 *   that calculates the nearest neighbor of each search point.
 * \param k The dimension size of the points
 * \param m The nubmer of search points
 * \param n The number of reference points
 * \param searchPoints
 * \param referencePoints
 * \param output
 * \return void. Results will be put in output
 */
__global__ void kernel(int k, int m, int n, float* searchPoints, float* referencePoints, int* output) {
    // int pid = threadIdx.x;
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int minIdx;
    float minSquareSum = -1;
    float diff, squareSum;
    // Iterate over all reference points
    if (pid < m) {
        for (int nInd = 0; nInd < n; nInd++) { // ref points
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) { // dimension
                diff = searchPoints[k * pid + kInd]
                     - referencePoints[k * nInd + kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIdx = nInd;
            }
        }
        output[pid] = minIdx;
    }
}

/*!
 * Wrapper of the CUDA kernel
 *   used to be called in the main function
 * \param k The dimension size of the points
 * \param m The nubmer of search points
 * \param n The number of reference points
 * \param searchPoints
 * \param referencePoints
 * \param results
 * \return void. Results will be put in result.
 */
extern void cudaCallbackGPU_baseline(int k, int m, int n, float *searchPoints,
                                     float *referencePoints, int **results) {
    float *searchPoints_d, *referencePoints_d;
    int* output_d;

    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&searchPoints_d, sizeof(float)*m*k));
    CHECK(cudaMalloc((void **)&referencePoints_d, sizeof(float)*n*k));
    CHECK(cudaMalloc((void **)&output_d, sizeof(int)*m));
    CHECK(cudaMemcpy(searchPoints_d, searchPoints, sizeof(float)*m*k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(referencePoints_d, referencePoints, sizeof(float)*n*k, cudaMemcpyHostToDevice));

    // Invoke the device function
    kernel<<< 1,m >>>(k, m, n, searchPoints_d, referencePoints_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *results = (int *)malloc(sizeof(int)*m);
    assert(results != NULL);
    CHECK(cudaMemcpy(*results, output_d, sizeof(int)*m, cudaMemcpyDeviceToHost));

    // int *cpu_results;
    // cudaCallbackCPU(k, m, n, searchPoints, referencePoints, &cpu_results);
    // for (int i = 0; i < m; ++i)
    //     assert(cpu_results[i] == (*results)[i]);

    CHECK(cudaFree(searchPoints_d));
    CHECK(cudaFree(referencePoints_d));
    CHECK(cudaFree(output_d));
}

/*!
 * Core execution part of CUDA
 *   that calculates the nearest neighbor of each search point.
 * \param k The dimension size of the points
 * \param m The nubmer of search points
 * \param n The number of reference points
 * \param searchPoints
 * \param referencePoints
 * \param output
 * \return void. Results will be put in output
 */
__global__ void kernel_sharedmem(int k, int m, int n, float* searchPoints, float* referencePoints, int* output) {
    int bid = blockIdx.x; // each block, one search point
    int tid = threadIdx.x;
    int n_points = ((n % 1024 == 0) ? n / 1024 : n / 1024 + 1);
    int searchId = bid;
    int referenceId = tid; // [tid,tid+n_points]
    int minIdx;
    float diff, squareSum;
    __shared__ float s_mem[16];
    if (tid < k) {
        s_mem[tid] = searchPoints[k * searchId + referenceId];
    }
    __syncthreads();
    __shared__ float dist[1024];
    __shared__ int dist_idx[1024];
    float minSquareSum = -1;
    for (int i = 0; i < n_points; ++i) {
        squareSum = 0;
        int refId = referenceId * n_points + i;
        for (int kInd = 0; kInd < k; kInd++) { // dimension
            diff = s_mem[kInd] - referencePoints[k * refId + kInd];
            squareSum += (diff * diff);
        }
        if (minSquareSum < 0 || squareSum < minSquareSum) {
            minSquareSum = squareSum;
            minIdx = refId;
        }
    }
    dist[referenceId] = minSquareSum;
    dist_idx[referenceId] = minIdx;
    __syncthreads();
    if (referenceId == 0) {
        float minSquareSum = -1;
        for (int i = 0; i < 1024; ++i) {
            squareSum = dist[i];
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIdx = dist_idx[i];
            }
        }
        output[searchId] = minIdx;
    }
}

/*!
 * Wrapper of the CUDA kernel
 *   used to be called in the main function
 * \param k The dimension size of the points
 * \param m The nubmer of search points
 * \param n The number of reference points
 * \param searchPoints
 * \param referencePoints
 * \param results
 * \return void. Results will be put in result.
 */
extern void cudaCallbackGPU_sharedmem(int k, int m, int n, float *searchPoints,
                                      float *referencePoints, int **results) {
    float *searchPoints_d, *referencePoints_d;
    int* output_d;

    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&searchPoints_d, sizeof(float)*m*k));
    CHECK(cudaMalloc((void **)&referencePoints_d, sizeof(float)*n*k));
    CHECK(cudaMalloc((void **)&output_d, sizeof(int)*m));
    CHECK(cudaMemcpy(searchPoints_d, searchPoints, sizeof(float)*m*k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(referencePoints_d, referencePoints, sizeof(float)*n*k, cudaMemcpyHostToDevice));

    // Invoke the device function
    kernel_sharedmem<<< m, 1024 >>>(k, m, n, searchPoints_d, referencePoints_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *results = (int *)malloc(sizeof(int)*m);
    assert(results != NULL);
    CHECK(cudaMemcpy(*results, output_d, sizeof(int)*m, cudaMemcpyDeviceToHost));

    // int *cpu_results;
    // cudaCallbackCPU(k, m, n, searchPoints, referencePoints, &cpu_results);
    // for (int i = 0; i < m; ++i)
    //     assert(cpu_results[i] == (*results)[i]);

    CHECK(cudaFree(searchPoints_d));
    CHECK(cudaFree(referencePoints_d));
    CHECK(cudaFree(output_d));
}