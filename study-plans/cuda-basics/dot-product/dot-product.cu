#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    // Write code here
    __shared__ float partial_sums[256];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(int i = globalIdx; i < N; i+=blockDim.x * gridDim.x) {
        sum += A[i] * B[i];
    }

    // Write the partial sums to partial_sums.
    partial_sums[threadIdx.x] = sum;

    // We could have also used a partial_sums[threadIdx.x] += a[i]*b[i], but that would have accessed the global memory more. over that, we use a sum variable which is stored in registers.
    
    __syncthreads(); //  then synchronize the threads

    // Now we start the tree reduction across the shared memory.

    for (int stride = blockDim.x / 2 ; stride > 0; stride /= 2) { 
        //  now we collapse the search space into logarithmic complexity.
        // stride size becomes 8 -> 4 -> 2 -> 1 -> 0 (/2 keeps only the int part)

        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride]; // add the stride-based additions recursively.
        }
        //  Synchronize the threads so that you know these partial sums do not cause race conditions.
        __syncthreads();
    }

    // now when everything has been collapsed into the first entry as the individual sums of their partial sums, now add these total collated partial sums to get the total sum 
    if (threadIdx.x == 0) {
        atomicAdd(&result[0], partial_sums[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
