#include <cuda_runtime.h>
#include <math.h>

// Just a few simple optimizations to increase performance by a bit. 
__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    // Write code here
    float sum = 0.0f;
    int row = blockIdx.x;

    for(int i = 0; i < N; i++) {
        // Now we need to cover the columns.
        sum += input[row*N + i] * input[row*N + i];
    }
    sum /= N*1.0f;
    sum += eps;

    float rms = sqrtf(sum);

    for(int i = threadIdx.x ; i < N; i+= blockDim.x) {
        // Now we need to cover the columns.
        output[row*N + i] = input[row*N + i]/rms * gamma[i];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}