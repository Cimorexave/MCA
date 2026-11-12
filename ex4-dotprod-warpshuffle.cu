#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>


// 1(a) Shared Memory Approach
__global__ void kernel_1a_shared(const double* x, int N, double* out_sum, double* out_1norm, double* out_2norm, int* out_zeros) {
    __shared__ double s_sum[256];
    __shared__ double s_1norm[256];
    __shared__ double s_2norm[256];
    __shared__ int s_zeros[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    s_sum[tid] = 0; s_1norm[tid] = 0; s_2norm[tid] = 0; s_zeros[tid] = 0;
    
    if (i < N) {
        double val = x[i];
        s_sum[tid] = val;
        s_1norm[tid] = abs(val);
        s_2norm[tid] = val * val;
        s_zeros[tid] = (val == 0.0) ? 1 : 0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_1norm[tid] += s_1norm[tid + stride];
            s_2norm[tid] += s_2norm[tid + stride];
            s_zeros[tid] += s_zeros[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_sum, s_sum[0]);
        atomicAdd(out_1norm, s_1norm[0]);
        atomicAdd(out_2norm, s_2norm[0]);
        atomicAdd(out_zeros, s_zeros[0]);
    }
}

// 1(b) Warp Shuffles Only (Fixed 256 blocks, grid-stride)
__global__ void kernel_1b_shfl_fixed(const double* x, int N, double* out_sum, double* out_1norm, double* out_2norm, int* out_zeros) {
    double sum = 0, norm1 = 0, norm2 = 0;
    int zeros = 0;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        double val = x[i];
        sum += val;
        norm1 += abs(val);
        norm2 += val * val;
        zeros += (val == 0.0) ? 1 : 0;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        norm1 += __shfl_down_sync(0xffffffff, norm1, offset);
        norm2 += __shfl_down_sync(0xffffffff, norm2, offset);
        zeros += __shfl_down_sync(0xffffffff, zeros, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(out_sum, sum);
        atomicAdd(out_1norm, norm1);
        atomicAdd(out_2norm, norm2);
        atomicAdd(out_zeros, zeros);
    }
}

// 1(c) Warp Shuffles Only (One thread per entry, dynamically scaled grid)
__global__ void kernel_1c_shfl_1thread(const double* x, int N, double* out_sum, double* out_1norm, double* out_2norm, int* out_zeros) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0, norm1 = 0, norm2 = 0;
    int zeros = 0;
    
    if (i < N) {
        double val = x[i];
        sum = val;
        norm1 = abs(val);
        norm2 = val * val;
        zeros = (val == 0.0) ? 1 : 0;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        norm1 += __shfl_down_sync(0xffffffff, norm1, offset);
        norm2 += __shfl_down_sync(0xffffffff, norm2, offset);
        zeros += __shfl_down_sync(0xffffffff, zeros, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(out_sum, sum);
        atomicAdd(out_1norm, norm1);
        atomicAdd(out_2norm, norm2);
        atomicAdd(out_zeros, zeros);
    }
}

// timing and execution scaffolding

void reset_results(double* d_sum, double* d_1norm, double* d_2norm, int* d_zeros) {
    cudaMemset(d_sum, 0, sizeof(double));
    cudaMemset(d_1norm, 0, sizeof(double));
    cudaMemset(d_2norm, 0, sizeof(double));
    cudaMemset(d_zeros, 0, sizeof(int));
}

int main() {
    int max_N = 100000000; // 10^8
    double* h_x = new double[max_N];
    for (int i = 0; i < max_N; ++i) {
        h_x[i] = 1.0; // Simple initialization
    }

    double *d_x, *d_sum, *d_1norm, *d_2norm;
    int *d_zeros;
    cudaMalloc(&d_x, max_N * sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));
    cudaMalloc(&d_1norm, sizeof(double));
    cudaMalloc(&d_2norm, sizeof(double));
    cudaMalloc(&d_zeros, sizeof(int));
    cudaMemcpy(d_x, h_x, max_N * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double cublas_result = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<int> N_values = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    int threadsPerBlock = 256;

    std::cout << std::left << std::setw(12) << "N" 
              << std::setw(15) << "1(a) Shared" 
              << std::setw(15) << "1(b) Shfl Fixed" 
              << std::setw(15) << "1(c) Shfl 1Th" 
              << std::setw(15) << "1(d) CUBLAS" << "\n";
    std::cout << std::string(70, '-') << "\n";

    // Warmup
    reset_results(d_sum, d_1norm, d_2norm, d_zeros);
    kernel_1b_shfl_fixed<<<256, threadsPerBlock>>>(d_x, 1000, d_sum, d_1norm, d_2norm, d_zeros);
    cudaDeviceSynchronize();

    for (int N : N_values) {
        float time_a = 0, time_b = 0, time_c = 0, time_cublas = 0;

        // 1(a) Shared Memory
        int blocks_a = std::ceil(N / (float)threadsPerBlock);
        reset_results(d_sum, d_1norm, d_2norm, d_zeros);
        cudaEventRecord(start);
        kernel_1a_shared<<<blocks_a, threadsPerBlock>>>(d_x, N, d_sum, d_1norm, d_2norm, d_zeros);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_a, start, stop);

        // 1(b) Warp Shuffles Fixed
        int blocks_b = 256;
        reset_results(d_sum, d_1norm, d_2norm, d_zeros);
        cudaEventRecord(start);
        kernel_1b_shfl_fixed<<<blocks_b, threadsPerBlock>>>(d_x, N, d_sum, d_1norm, d_2norm, d_zeros);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_b, start, stop);

        // 1(c) Warp Shuffles 1 Thread per entry
        int blocks_c = std::ceil(N / (float)threadsPerBlock);
        reset_results(d_sum, d_1norm, d_2norm, d_zeros);
        cudaEventRecord(start);
        kernel_1c_shfl_1thread<<<blocks_c, threadsPerBlock>>>(d_x, N, d_sum, d_1norm, d_2norm, d_zeros);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_c, start, stop);

        // 1(d) CUBLAS Dot Product
        cudaEventRecord(start);
        cublasDdot(handle, N, d_x, 1, d_x, 1, &cublas_result);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_cublas, start, stop);

        std::cout << std::left << std::setw(12) << N 
                  << std::setw(15) << time_a 
                  << std::setw(15) << time_b 
                  << std::setw(15) << time_c 
                  << std::setw(15) << time_cublas << "\n";
    }

    cublasDestroy(handle);
    cudaFree(d_x); cudaFree(d_sum); cudaFree(d_1norm); cudaFree(d_2norm); cudaFree(d_zeros);
    delete[] h_x;

    return 0;
}