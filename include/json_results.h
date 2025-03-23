#ifndef JSON_RESULTS_H
#define JSON_RESULTS_H

#define MAX_CUDA_CSR_KERNELS 10
#define MAX_CUDA_HLL_KERNELS 10
#define MAX_OPENMP_RESULTS   50

// Struttura generica di performance
typedef struct {
    double time;
    double flops;
} PerfRecord;

// Struttura per risultati openmp
typedef struct {
    int threads;
    double time;
    double flops;
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
    char matrix_name[256];  // nome della matrice

    // Sezione "serial" (un record con time e flops)
    PerfRecord serial;

    // Sezione "cuda" (un record se ne vuoi uno generico)
    PerfRecord cuda;

    // Se vuoi salvare TUTTI i kernel CSR
    CudaCSRKernelPerf cuda_csr[MAX_CUDA_CSR_KERNELS];
    int num_cuda_csr;

    // Se vuoi salvare TUTTI i kernel HLL
    CudaHLLKernelPerf cuda_hll[MAX_CUDA_HLL_KERNELS];
    int num_cuda_hll;

    // Sezione "openmp"
    OpenMPPerf openmp_results[MAX_OPENMP_RESULTS];
    int num_openmp;
} MatrixResult;

// Funzione di scrittura JSON
void write_results_to_json(const char *filename, MatrixResult *results, int num_matrices);

#endif
