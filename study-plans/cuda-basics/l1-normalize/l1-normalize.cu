__global__ void l1_reduce_kernel(const float* input, float* norm, int N) {
    __shared__ float partial[256];

    float sum = 0.0f;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = globalIdx; i < N; i += blockDim.x * gridDim.x) {
        sum += fabsf(input[i]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(norm, partial[0]);
}

__global__ void l1_divide_kernel(const float* input, float* output, const float* norm, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float n = *norm;
        output[i] = (n > 0.0f) ? (input[i] / n) : 0.0f;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_norm;
    cudaMalloc(&d_norm, sizeof(float));
    cudaMemset(d_norm, 0, sizeof(float));

    l1_reduce_kernel<<<blocks, threads>>>(input, d_norm, N);
    l1_divide_kernel<<<blocks, threads>>>(input, output, d_norm, N);

    cudaFree(d_norm);
    cudaDeviceSynchronize();
}