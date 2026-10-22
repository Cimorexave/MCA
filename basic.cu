#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
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

struct HostTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

__global__ void init_vectors_kernel(double *a, double *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<double>(idx);
        b[idx] = static_cast<double>(n - 1 - idx);
    }
}

__global__ void vector_add_kernel(const double *a, const double *b, double *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_grid_stride_kernel(const double *a, const double *b, double *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

void task_a() {
    std::cout << "Task a)" << std::endl;
    std::cout << std::setw(12) << "N"
              << std::setw(16) << "Size (MB)"
              << std::setw(18) << "cudaMalloc (ms)"
              << std::setw(16) << "cudaFree (ms)" << std::endl;

    std::vector<size_t> test_sizes = {1000, 10000, 100000, 1000000, 10000000};
    HostTimer timer;

    for (size_t n : test_sizes) {
        size_t bytes = n * sizeof(double);
        double *d_ptr = nullptr;

        timer.start();
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, bytes));
        double malloc_time = timer.stop_ms();

        timer.start();
        CUDA_CHECK(cudaFree(d_ptr));
        double free_time = timer.stop_ms();

        std::cout << std::setw(12) << n
                  << std::setw(16) << (bytes / (1024.0 * 1024.0))
                  << std::setw(18) << malloc_time
                  << std::setw(16) << free_time << std::endl;
    }
}

void task_b() {
    std::cout << "Task b)" << std::endl;

    const size_t n = 100000;
    const size_t bytes = n * sizeof(double);
    const double total_mb = (2.0 * bytes) / (1024.0 * 1024.0);

    double *d_a = nullptr;
    double *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));

    HostTimer timer;

    // direct GPU kernel initialization
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    timer.start();
    init_vectors_kernel<<<grid_size, block_size>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    double time_opt1 = timer.stop_ms();
    double bw_opt1 = (total_mb) / (time_opt1 / 1000.0);

    // bulk host copy wtih cudaMemcpy
    std::vector<double> h_a(n), h_b(n);
    for (size_t i = 0; i < n; ++i) {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(n - 1 - i);
    }

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    double time_opt2 = timer.stop_ms();
    double bw_opt2 = (total_mb) / (time_opt2 / 1000.0);

    // element by element cudaMemcpy
    timer.start();
    for (size_t i = 0; i < n; ++i) {
        CUDA_CHECK(cudaMemcpy(&d_a[i], &h_a[i], sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_b[i], &h_b[i], sizeof(double), cudaMemcpyHostToDevice));
    }
    double time_opt3 = timer.stop_ms();
    double bw_opt3 = (total_mb) / (time_opt3 / 1000.0);

    std::cout << "Data transferred / initialized: " << total_mb << " MB\n";
    std::cout << " Dedicated GPU Kernel:  Time = " << std::setw(10) << time_opt1 
              << " ms | Effective BW = " << bw_opt1 << " MB/s\n";
    std::cout << " Bulk Host Copy:        Time = " << std::setw(10) << time_opt2 
              << " ms | Effective BW = " << bw_opt2 << " MB/s\n";
    std::cout << " Element By Element:    Time = " << std::setw(10) << time_opt3 
              << " ms | Effective BW = " << bw_opt3 << " MB/s\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

void task_cd() {
    std::cout << "Task c and d)" << std::endl;
    std::cout << std::setw(12) << "N" 
              << std::setw(20) << "Execution Time (ms)" << std::endl;

    std::vector<size_t> sizes = {
        100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000
    };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int block_size = 256;

    for (size_t n : sizes) {
        size_t bytes = n * sizeof(double);
        double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

        int grid_size = (n + block_size - 1) / block_size;
        init_vectors_kernel<<<grid_size, block_size>>>(d_a, d_b, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        std::cout << std::setw(12) << n << std::setw(20) << elapsed_ms << std::endl;

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void task_e() {
    std::cout << "Task e)" << std::endl;

    const size_t n = 3000000;
    const size_t bytes = n * sizeof(double);
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));
    init_vectors_kernel<<<grid_size, block_size>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int run = 1; run <= 5; ++run) {
        CUDA_CHECK(cudaEventRecord(start));
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "Run " << run << ": " << elapsed_ms << " ms" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void task_f() {
    std::cout << "Task f)" << std::endl;

    const size_t n = 10000000;
    const size_t bytes = n * sizeof(double);

    double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    int base_block = 256;
    int base_grid = (n + base_block - 1) / base_block;
    init_vectors_kernel<<<base_grid, base_block>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> test_values = {16, 32, 64, 128, 256, 512, 1024};
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Vary Block Size (fixed grid covering all elements)
    std::cout << "\n--- Fixed Grid Coverage, Varying Block Size ---" << std::endl;
    std::cout << std::setw(15) << "Block Size" << std::setw(20) << "Time (ms)" << std::endl;
    for (int bs : test_values) {
        int gs = (n + bs - 1) / bs;
        CUDA_CHECK(cudaEventRecord(start));
        vector_add_kernel<<<gs, bs>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << std::setw(15) << bs << std::setw(20) << elapsed_ms << std::endl;
    }

    // Vary Grid Size with Grid-Stride Loop (fixed block size = 256)
    std::cout << "\n--- Fixed Block Size (256), Varying Grid Size (Grid-Stride) ---" << std::endl;
    std::cout << std::setw(15) << "Grid Size" << std::setw(20) << "Time (ms)" << std::endl;
    for (int gs : test_values) {
        CUDA_CHECK(cudaEventRecord(start));
        vector_add_grid_stride_kernel<<<gs, 256>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << std::setw(15) << gs << std::setw(20) << elapsed_ms << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    // Warm up runtime and initialize device context
    CUDA_CHECK(cudaFree(0));

    task_a();
    task_b();
    task_cd();
    task_e();
    task_f();

    return 0;
}