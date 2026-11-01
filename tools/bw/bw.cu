// Host to device copy speed: pageable vs pinned, for the copy sizes the MoE cache uses.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { if ((x) != cudaSuccess) { printf("fail %s\n", #x); exit(1); } } while (0)
static double run(char * host, char * dev, size_t total, size_t chunk) {
    int n = total / chunk;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < n; i++) CK(cudaMemcpyAsync(dev + i*chunk, host + i*chunk, chunk, cudaMemcpyHostToDevice, 0));
        CK(cudaStreamSynchronize(0));
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return 4.0*n*chunk/1e9/s;
}
int main() {
    size_t total = 2ull << 30;
    char * dev; CK(cudaMalloc(&dev, total));
    char * pageable = (char *) aligned_alloc(4096, total); memset(pageable, 1, total);
    char * pinned; CK(cudaMallocHost(&pinned, total)); memset(pinned, 1, total);
    for (size_t c : {64ull<<10, 512ull<<10, 2ull<<20, 8ull<<20, 64ull<<20}) {
        printf("chunk %6zu KiB  pageable %6.2f GB/s  pinned %6.2f GB/s\n", c>>10, run(pageable, dev, total, c), run(pinned, dev, total, c));
    }
    return 0;
}
