#include <cuda_runtime.h>
#include<math.h>
#include<vector_types.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    /* relu(x) = { 0 if x<0
                 { x if x>0 */
    // converting float to float4 to give alignment of 16 byte
  const  float4* vec_in = reinterpret_cast<const float4*>(input);
        float4* vec_ou = reinterpret_cast<float4*>(output);
   
    // vectroised limit
    int N_vec = N/4;

    int stride = blockDim.x * gridDim.x;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int i=tx;i<N_vec;i+=stride){
        float4 vec = vec_in[tx]; // load 16 byte into vec from vec_in array index = tx
        vec.x = fmaxf(vec.x,0.0f);
         vec.y = fmaxf(vec.y,0.0f);
         vec.z = fmaxf(vec.z,0.0f);
         vec.w = fmaxf(vec.w,0.0f);
        vec_ou[tx] = vec; //  store 16 byte into vec_ou from vec at array index = tx
    }

    int remainder = N_vec * 4 + tx;
        for(int i=remainder;i<N;i+=stride){
            output[i]= fmaxf(input[i],0.0f);
        }
    
    
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256; 
    int vec = N/4 ; 
    int blocks = fmaxf(1,(vec + threads - 1) / threads);
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}