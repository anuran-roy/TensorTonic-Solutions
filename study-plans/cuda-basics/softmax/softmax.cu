#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    float sum = 0.0f;
    float max_val = 0.0f;

    for (int i=0; i<N; i++) {
        max_val = max(max_val, input[i]);
    }
    
    for (int i = 0; i<N; i++) {
        sum += exp(input[i] - max_val);
    }

    for (int i = 0; i<N; i++) {
        output[i] = exp(input[i] - max_val)/sum;
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}