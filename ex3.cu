#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void vectorSumStrided(const float* x, const float* y, float* z, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / k) z[i * k] = x[i * k] + y[i * k];
}

__global__ void vectorSumOffset(const float* x, const float* y, float* z, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - k) z[i + k] = x[i + k] + y[i + k];
}

void runTask1() {
    std::cout << "\n--- TASK 1: Strided and Offset Memory Access ---\n";
    int N = 10000000;
    size_t bytes = N * sizeof(float);
    float *d_x, *d_y, *d_z;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_y, bytes));
    CHECK_CUDA(cudaMalloc(&d_z, bytes));

    int threads = 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "k\tStrided BW (GB/s)\tOffset BW (GB/s)\n";
    for (int k = 1; k <= 63; ++k) {
        int blocksStrided = (N / k + threads - 1) / threads;
        cudaEventRecord(start);
        vectorSumStrided<<<blocksStrided, threads>>>(d_x, d_y, d_z, N, k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float msStrided;
        cudaEventElapsedTime(&msStrided, start, stop);
        
        double bytesStrided = 3.0 * (N / k) * sizeof(float);
        double bwStrided = (bytesStrided / (msStrided / 1000.0)) / 1e9;

        int blocksOffset = (N - k + threads - 1) / threads;
        cudaEventRecord(start);
        vectorSumOffset<<<blocksOffset, threads>>>(d_x, d_y, d_z, N, k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float msOffset;
        cudaEventElapsedTime(&msOffset, start, stop);
        
        double bytesOffset = 3.0 * (N - k) * sizeof(float);
        double bwOffset = (bytesOffset / (msOffset / 1000.0)) / 1e9;

        if (k == 1 || k % 8 == 0 || k == 63) {
            std::cout << k << "\t" << bwStrided << "\t\t\t" << bwOffset << "\n";
        }
    }
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
}

__global__ void transposeInPlaceFixed(float* A, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N && x > y) {
        float temp = A[y * N + x];
        A[y * N + x] = A[x * N + y];
        A[x * N + y] = temp;
    }
}

#define TILE_DIM 16
__global__ void transposeInPlaceShared(float* A, int N) {
    __shared__ float tileA[TILE_DIM][TILE_DIM+1];
    __shared__ float tileB[TILE_DIM][TILE_DIM+1];
    int bx = blockIdx.x * TILE_DIM, by = blockIdx.y * TILE_DIM;
    int tx = threadIdx.x, ty = threadIdx.y;

    if (bx >= by) {
        int x1 = bx + tx, y1 = by + ty;
        int x2 = by + tx, y2 = bx + ty;

        if (x1 < N && y1 < N) tileA[ty][tx] = A[y1 * N + x1];
        if (x2 < N && y2 < N) tileB[ty][tx] = A[y2 * N + x2];
        __syncthreads();

        if (x2 < N && y2 < N) A[y2 * N + x2] = tileA[tx][ty];
        if (bx != by && x1 < N && y1 < N) A[y1 * N + x1] = tileB[tx][ty];
    }
}

void runTask2() {
    std::cout << "\n--- TASK 2: Dense Matrix Transpose ---\n";
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "N\tFixed BW (GB/s)\t\tShared BW (GB/s)\n";
    for (int N : sizes) {
        size_t bytes = N * N * sizeof(float);
        float* d_A;
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

        cudaEventRecord(start);
        transposeInPlaceFixed<<<blocks, threads>>>(d_A, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float msFixed;
        cudaEventElapsedTime(&msFixed, start, stop);
        double bwFixed = (2.0 * bytes / (msFixed / 1000.0)) / 1e9; // Read and write

        cudaEventRecord(start);
        transposeInPlaceShared<<<blocks, threads>>>(d_A, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float msShared;
        cudaEventElapsedTime(&msShared, start, stop);
        double bwShared = (2.0 * bytes / (msShared / 1000.0)) / 1e9;

        std::cout << N << "\t" << bwFixed << "\t\t" << bwShared << "\n";
        cudaFree(d_A);
    }
}

__global__ void emptyKernel() {}

__global__ void copyKernel(float* out, const float* in, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx];
}

__global__ void atomicKernel(int* val, int iters) {
    for(int i=0; i<iters; ++i) atomicAdd(val, 1);
}

__global__ void flopKernel(float* out, int iters) {
    float a = threadIdx.x * 0.1f, b = 1.01f, c = 1.02f;
    for (int i = 0; i < iters; i++) {
        a = a * b + c; a = a * b + c; a = a * b + c; a = a * b + c;
        a = a * b + c; a = a * b + c; a = a * b + c; a = a * b + c;
        a = a * b + c; a = a * b + c; a = a * b + c; a = a * b + c;
        a = a * b + c; a = a * b + c; a = a * b + c; a = a * b + c;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

void runTask3() {
    std::cout << "\n--- task 3: Performance Modeling ---\n";
    
    char h_byte = 1, *d_byte;
    cudaMalloc(&d_byte, 1);
    int iters = 10000;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) {
        cudaMemcpy(d_byte, &h_byte, 1, cudaMemcpyHostToDevice);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double pcieLat = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)iters;
    std::cout << "(a) PCIe Latency: " << pcieLat << " microseconds\n";
    cudaFree(d_byte);

    t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) {
        emptyKernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    double launchLat = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)iters;
    std::cout << "(b) Kernel Launch Latency: " << launchLat << " microseconds\n";

    size_t bwN = 64 * 1024 * 1024; 
    float *d_in, *d_out;
    cudaMalloc(&d_in, bwN * sizeof(float)); cudaMalloc(&d_out, bwN * sizeof(float));
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    copyKernel<<<(bwN + 255)/256, 256>>>(d_out, d_in, bwN);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msCopy; cudaEventElapsedTime(&msCopy, start, stop);
    double peakBw = (2.0 * bwN * sizeof(float) / (msCopy / 1000.0)) / 1e9;
    std::cout << "(c) Practical Peak Memory BW: " << peakBw << " GB/sec\n";
    cudaFree(d_in); cudaFree(d_out);

    int *d_val;
    cudaMalloc(&d_val, sizeof(int));
    cudaMemset(d_val, 0, sizeof(int));
    int atomicIters = 1000;
    int threads = 1024, blocks = 256;
    cudaEventRecord(start);
    atomicKernel<<<blocks, threads>>>(d_val, atomicIters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msAtomic; cudaEventElapsedTime(&msAtomic, start, stop);
    double totalAtomics = (double)blocks * threads * atomicIters;
    std::cout << "(d) Atomic Updates/sec: " << (totalAtomics / (msAtomic / 1000.0)) << "\n";
    cudaFree(d_val);

    float *d_flopOut;
    cudaMalloc(&d_flopOut, blocks * sizeof(float));
    int flopIters = 100000; 
    cudaEventRecord(start);
    flopKernel<<<blocks, threads>>>(d_flopOut, flopIters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msFlop; cudaEventElapsedTime(&msFlop, start, stop);
    double totalFlops = (double)blocks * threads * flopIters * 32.0; // 32 ops per iteration
    std::cout << "(e) Peak Single Precision GFLOPs: " << (totalFlops / (msFlop / 1000.0)) / 1e9 << " GFLOPs/sec\n";
    cudaFree(d_flopOut);
}

int main() {
    runTask1();
    runTask2();
    runTask3();
    return 0;
}