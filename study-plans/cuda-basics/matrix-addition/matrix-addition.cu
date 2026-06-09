#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    // Write code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                C[row * N + col] = A[row * N + col] + B[row * N + col];
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
