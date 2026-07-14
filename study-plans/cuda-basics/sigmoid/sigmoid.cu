#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    int tx= blockDim.x * blockIdx.x + threadIdx.x; 
    const float4* vec_in = reinterpret_cast<const float4*>(input);
    float4* vec_oup = reinterpret_cast<float4*>(output);
int N_vec = N/4;
    float4 add;
    int strid = blockDim.x * gridDim.x;
     float4 div;
    for(int i=tx;i<N_vec;i+=strid){
        float4 vec = vec_in[i];
        add.x = (expf(vec.x))+1;
        add.y = (expf(vec.y))+1;
        add.z = (expf(vec.z))+1;
        add.w = (expf(vec.w))+1;
   
        div.x = expf(vec.x)/add.x;
          div.y = expf(vec.y)/add.y;
          div.z = expf(vec.z)/add.z;
          div.w = expf(vec.w)/add.w;
       vec_oup[i]=div;
    }
    // processing rest thread
    int remain= N_vec *4 +tx;
   float add2;
    for(int i=remain;i<N;i+=strid){
        add2 = (expf(input[i]))+1;
        output[i] = expf(input[i])/add2;
    }
    
       
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int vec = N/4;
    int blocks = fmaxf(1,(vec + threads - 1)/threads);
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}