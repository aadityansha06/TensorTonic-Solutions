#include <cuda_runtime.h>
#include<math.h>

__global__ void relu_kernel(const float* input, float* output, int N) {

    /* relu(x) = { 0 if x<0
                 { x if x>0
    */
    
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx<N){
            output[tx] = fmaxf(input[tx],0); // max of input or 0 will go 
                                                
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}