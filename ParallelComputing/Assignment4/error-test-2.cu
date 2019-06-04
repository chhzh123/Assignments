#include <cstdio>
#include <cmath>

__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    // Add the kernel code
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 20;
    const int ThreadsInBlock = 128;
    double *dA, *dB, *dC;
    double hA[N], hB[N], hC[N];
  
    for(int i = 0; i < N; ++i) {
        hA[i] = (double) i;
        hB[i] = (double) i * i;
    }


    cudaMalloc((void**)&dA, sizeof(double)*N);
    // #error Add the remaining memory allocations and copies
    cudaMalloc((void**)&dB, sizeof(double)*N);
    cudaMalloc((void**)&dC, sizeof(double)*N);

    cudaMemcpy((void*)dA,(void*)hA,N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dB,(void*)hB,N*sizeof(double),cudaMemcpyHostToDevice);

    // Note the maximum size of threads in a block
    dim3 grid, threads;

    //// Add the kernel call here
    // #error Add the CUDA kernel call
    vector_add <<<1,ThreadsInBlock>>> (dC,dA,dB,N);

    // error: access host memory
    // vector_add <<<1,ThreadsInBlock>>> (hC,hA,hB,N);

    //// Copy back the results and free the device memory
    // #error Copy back the results and free the allocated memory
    cudaMemcpy((void*)hC,(void*)dC,N*sizeof(double),cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++)
    //     printf("%5.1f\n", hC[i]);

    for (int i = 0; i < N; i++)
        printf("%5.1f\n", hC[i]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}