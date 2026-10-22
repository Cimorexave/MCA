#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

constexpr int NUM_BLOCKS = 256;
constexpr int THREADS_PER_BLOCK = 256;

__global__ void init_vectors(double *a, double *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        a[i] = static_cast<double>(i);
        b[i] = static_cast<double>(n - 1 - i);
    }
}

__global__ void dot_product_stage1(const double *a, const double *b, double *partial_sums, size_t n) {
    __shared__ double sdata[THREADS_PER_BLOCK];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    double thread_sum = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        thread_sum += a[i] * b[i];
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void dot_product_stage2(const double *partial_sums, double *final_result, int num_elements) {
    __shared__ double sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    sdata[tid] = (tid < num_elements) ? partial_sums[tid] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *final_result = sdata[0];
    }
}

void run_dot_product_test(size_t n) {
    size_t bytes_vec = n * sizeof(double);
    size_t bytes_partial = NUM_BLOCKS * sizeof(double);

    double *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr, *d_final = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes_vec));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes_vec));
    CUDA_CHECK(cudaMalloc((void**)&d_partial, bytes_partial));
    CUDA_CHECK(cudaMalloc((void**)&d_final, sizeof(double)));

    init_vectors<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dot_product_stage1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial, n);
    dot_product_stage2<<<1, NUM_BLOCKS>>>(d_partial, d_final, NUM_BLOCKS);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    dot_product_stage1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial, n);
    dot_product_stage2<<<1, NUM_BLOCKS>>>(d_partial, d_final, NUM_BLOCKS);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_gpu_ms, start, stop));

    double res_a = 0.0;
    CUDA_CHECK(cudaMemcpy(&res_a, d_final, sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_partial(NUM_BLOCKS);
    double res_b = 0.0;

    CUDA_CHECK(cudaEventRecord(start));

    dot_product_stage1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial, n);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, bytes_partial, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_BLOCKS; ++i) {
        res_b += h_partial[i];
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_hybrid_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_hybrid_ms, start, stop));

    double expected = (static_cast<double>(n) * (static_cast<double>(n) * n - 1.0)) / 6.0;

    std::cout << "N = " << std::setw(10) << n
              << " | (a) Pure GPU: " << std::setw(8) << time_gpu_ms << " ms"
              << " | (b) GPU+CPU: " << std::setw(8) << time_hybrid_ms << " ms"
              << " | Result Diff: " << std::abs(res_a - expected) << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_final));
}

int main() {
    CUDA_CHECK(cudaFree(0));
    std::cout << "Dot Product Performance Comparison (256 Blocks)\n";

    std::vector<size_t> test_sizes = {1000, 10000, 100000, 1000000, 10000000};
    for (size_t n : test_sizes) {
        run_dot_product_test(n);
    }

    return 0;
}