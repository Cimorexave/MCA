#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <iomanip>

// ==============================================================================
// ChatGPT Sequential Implementation (using 2D array of pointers)
// ==============================================================================
__global__ void kernel_chatgpt_multidot(const double* x, double** y, int N, int K, double* out) {
    int k = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    if (i < N && k < K) {
        sum = x[i] * y[k][i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0 && k < K) {
        atomicAdd(&out[k], sum);
    }
}

// ==============================================================================
// 2(c) Specialized 8-way kernel
// ==============================================================================
__global__ void kernel_2c_multidot8(const double* x, 
                                    const double* y0, const double* y1, const double* y2, const double* y3,
                                    const double* y4, const double* y5, const double* y6, const double* y7,
                                    int N, double* out) {
    double sums[8] = {0};
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        double xv = x[i];
        sums[0] += xv * y0[i];
        sums[1] += xv * y1[i];
        sums[2] += xv * y2[i];
        sums[3] += xv * y3[i];
        sums[4] += xv * y4[i];
        sums[5] += xv * y5[i];
        sums[6] += xv * y6[i];
        sums[7] += xv * y7[i];
    }

    for (int k = 0; k < 8; ++k) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sums[k] += __shfl_down_sync(0xffffffff, sums[k], offset);
        }
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&out[k], sums[k]);
        }
    }
}

int main() {
    int max_N = 1000000;
    int max_K = 32;

    double* d_x;
    cudaMalloc(&d_x, max_N * sizeof(double));

    // Allocate K vectors for y
    std::vector<double*> h_y_ptrs(max_K);
    for (int k = 0; k < max_K; ++k) {
        cudaMalloc(&h_y_ptrs[k], max_N * sizeof(double));
    }

    // Array of pointers on device (for ChatGPT version)
    double** d_y_ptrs;
    cudaMalloc(&d_y_ptrs, max_K * sizeof(double*));
    cudaMemcpy(d_y_ptrs, h_y_ptrs.data(), max_K * sizeof(double*), cudaMemcpyHostToDevice);

    double* d_out;
    cudaMalloc(&d_out, max_K * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<int> N_values = {1000, 10000, 100000, 1000000};
    std::vector<int> K_values = {8, 16, 24, 32};

    std::cout << std::left << std::setw(10) << "N" 
              << std::setw(10) << "K" 
              << std::setw(20) << "ChatGPT (ms)" 
              << std::setw(20) << "8-Way Custom (ms)" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int N : N_values) {
        for (int K : K_values) {
            float time_chatgpt = 0, time_custom = 0;

            // Run ChatGPT Version
            cudaMemset(d_out, 0, K * sizeof(double));
            dim3 blocks(std::ceil(N / 256.0), K);
            cudaEventRecord(start);
            kernel_chatgpt_multidot<<<blocks, 256>>>(d_x, d_y_ptrs, N, K, d_out);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_chatgpt, start, stop);

            // Run 8-Way Custom Version
            cudaMemset(d_out, 0, K * sizeof(double));
            cudaEventRecord(start);
            for (int k = 0; k < K; k += 8) {
                kernel_2c_multidot8<<<256, 256>>>(
                    d_x, 
                    h_y_ptrs[k], h_y_ptrs[k+1], h_y_ptrs[k+2], h_y_ptrs[k+3],
                    h_y_ptrs[k+4], h_y_ptrs[k+5], h_y_ptrs[k+6], h_y_ptrs[k+7],
                    N, d_out + k
                );
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_custom, start, stop);

            std::cout << std::left << std::setw(10) << N 
                      << std::setw(10) << K 
                      << std::setw(20) << time_chatgpt 
                      << std::setw(20) << time_custom << "\n";
        }
    }

    // Cleanup
    cudaFree(d_x);
    for (int k = 0; k < max_K; ++k) cudaFree(h_y_ptrs[k]);
    cudaFree(d_y_ptrs);
    cudaFree(d_out);

    return 0;
}