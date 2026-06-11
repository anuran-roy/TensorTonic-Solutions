#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    // Write code here

    // Initialize the shared memory space first.
    __shared__ float partial_sums[256];

    float sum = 0.0f;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    partial_sums[threadIdx.x] = sum;

    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }

    __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result[0], partial_sums[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
