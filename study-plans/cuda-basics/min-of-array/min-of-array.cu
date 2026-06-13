#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* partial, int N) {
    // Write code here
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float partial_maxes[256];

    float min_val = FLT_MAX;

    for(int i = globalIdx; i < N; i += blockDim.x * gridDim.x) {
        min_val = fminf(min_val, input[i]);
    }

    partial_maxes[threadIdx.x] = min_val;
    __syncthreads();

    // Now we go to tree reduction

    for (int stride = blockDim.x/2 ; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_maxes[threadIdx.x] = fminf(partial_maxes[threadIdx.x], partial_maxes[threadIdx.x + stride]);
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = partial_maxes[0];
    }
}

__global__ void final_min_kernel(const float* partial, float* result, int blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    sdata[tid] = (tid < blocks) ? partial[tid] : FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) *result = sdata[0];
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    min_kernel<<<blocks, threads>>>(input, d_partial, N);
    final_min_kernel<<<1, 256>>>(d_partial, result, blocks);

    cudaDeviceSynchronize();
    cudaFree(d_partial);
}
