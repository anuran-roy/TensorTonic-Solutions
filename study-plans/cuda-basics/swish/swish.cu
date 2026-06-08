#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = input[i]/(1 + exp(-1.0 * input[i]));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
