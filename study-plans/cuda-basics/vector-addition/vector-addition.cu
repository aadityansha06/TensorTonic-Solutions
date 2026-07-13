#include <cuda_runtime.h>
#include<stdio.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if(tx<N){
        C[tx]=A[tx]+B[tx];
    }
    
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}