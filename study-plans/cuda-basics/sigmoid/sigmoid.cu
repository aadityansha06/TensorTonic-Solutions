#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    int tx= blockDim.x * blockIdx.x + threadIdx.x;   
    float add;
    if(tx<N){
       // input[tx] = input[tx] * -1;
        add = (expf(input[tx]))+1;
        output[tx]=expf(input[tx])/add;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}