#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    // Write code here

    __shared__ float partial_sums_regular[256];
    __shared__ float partial_sums_square[256];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    float regular_sum = 0.0f;
    float squares_sum = 0.0f;

    for(int i = globalIdx; i < N; i += blockDim.x * gridDim.x) {
        regular_sum += input[i];
        squares_sum += input[i] * input[i];
    }

    partial_sums_regular[threadIdx.x] = regular_sum;
    partial_sums_square[threadIdx.x] = squares_sum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride){
            partial_sums_regular[threadIdx.x] += partial_sums_regular[threadIdx.x + stride];
            partial_sums_square[threadIdx.x] += partial_sums_square[threadIdx.x + stride];
        }

        __syncthreads();
    }


    if (threadIdx.x == 0) {
        atomicAdd(mean_out, partial_sums_regular[0]);
        atomicAdd(var_out, partial_sums_square[0]);
    }
    
}

__global__ void finalize_kernel(float* mean_out, float* var_out, int N) {
    float mean = *mean_out / N;
    float var  = *var_out  / N - mean * mean;
    *mean_out = mean;
    *var_out  = var;
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    mean_variance_kernel<<<blocks, threads>>>(input, mean_out, var_out, N);
    finalize_kernel<<<1, 1>>>(mean_out, var_out, N);
    cudaDeviceSynchronize();
}
