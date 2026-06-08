#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float exp1 = exp(input[i]);
        float exp2 = exp(-1.0 * input[i]);
        
        output[i] = (exp1 - exp2)/(exp1 + exp2);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}