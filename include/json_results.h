#ifndef JSON_RESULTS_H
#define JSON_RESULTS_H

#define MAX_CUDA_CSR_KERNELS 10
#define MAX_CUDA_HLL_KERNELS 10
#define MAX_OPENMP_CSR       50
#define MAX_OPENMP_HLL       50

// Struttura generica di performance
typedef struct {
    double time;
    double flops;
} PerfRecord;

// Struttura per risultati OpenMP
typedef struct {
    int threads;
    double time;
    double flops;
    double speedup;
    double efficienza;
} OpenMPPerf;

// Kernel CSR
typedef struct {
    char kernel_name[64];
    double time;
    double gflops;
} CudaCSRKernelPerf;

// Kernel HLL
typedef struct {
    char kernel_name[64];
    double time;
    double gflops;
} CudaHLLKernelPerf;

// Struttura principale
typedef struct {
    char matrix_name[256];  // Nome della matrice

    // Sezione "serial" (un record con time e flops)
    PerfRecord serial;

    // Sezione "cuda" (un record se ne vuoi uno generico)
    PerfRecord cuda;

    // Kernel CUDA CSR
    CudaCSRKernelPerf cuda_csr[MAX_CUDA_CSR_KERNELS];
    int num_cuda_csr;

    // Kernel CUDA HLL
    CudaHLLKernelPerf cuda_hll[MAX_CUDA_HLL_KERNELS];
    int num_cuda_hll;

    // OpenMP CSR
    OpenMPPerf openmp_csr[MAX_OPENMP_CSR];
    int num_openmp_csr;

    // OpenMP HLL
    OpenMPPerf openmp_hll[MAX_OPENMP_HLL];
    int num_openmp_hll;

} MatrixResult;

// Funzione di scrittura JSON
void write_results_to_json(const char *filename, MatrixResult *results, int num_matrices);

#endif
