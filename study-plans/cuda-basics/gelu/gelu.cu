#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write code here
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        output[row] = 0.5 * input[row] * (1 + tanh(sqrt(2/M_PI) * (input[row] + 0.044715 * pow(input[row], 3.0))));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
