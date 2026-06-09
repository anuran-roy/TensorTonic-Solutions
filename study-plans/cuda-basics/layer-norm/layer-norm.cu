#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    // Write code here
    int row = blockIdx.x;

    if (row < M) {
        float mu = 0.0f;
        float variance = 0.0f;
        for(int i = 0; i < N; i++) {
            mu += input[row * N + i];
        }

        mu /= N;

        for (int i = 0; i < N; i++) {
            variance += pow(input[row * N + i] - mu, 2);
        }

        variance /= N;
        float inv_std = rsqrtf(variance + eps);

        for (int col = threadIdx.x; col < N; col += blockDim.x) {
            output[row * N + col] = (input[row * N + col] - mu) * inv_std * gamma[col] + beta[col];
        }
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
