#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    // Write code here
    __shared__ float partial_sums[256];
    float sum = 0.0f;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = globalIdx; i < N; i += blockDim.x * gridDim.x ) {
        sum += input[i] * input[i];
    };

    partial_sums[threadIdx.x] = sum;

    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }

    __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sumv, partial_sums[0]);
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    // Write code here
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        output[row] = rsqrtf(*sumv) * input[row];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
