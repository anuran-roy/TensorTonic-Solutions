#include <cuda_runtime.h>

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    // Write code here
    int row = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M) {
        y[row] = 0;

        for (int i=0; i < N; i++) {
            y[row] += A[row * N + i] * x[i];
        }
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(256);
    dim3 blocks((M + 255) / 256);
    gemv_kernel<<<blocks, threads>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
