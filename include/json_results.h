#ifndef JSON_RESULTS_H
#define JSON_RESULTS_H

#define MAX_OPENMP_CONFIGS 6

typedef struct {
    int threads;
    double time;
    double flops;
} OpenMPResult;

typedef struct {
    double time;
} SerialResult;

typedef struct {
    double time;
    double flops;
} CudaResult;

typedef struct {
    char matrix_name[256];
    SerialResult serial;
    CudaResult cuda;
    OpenMPResult openmp_results[MAX_OPENMP_CONFIGS];
    int num_openmp;
} MatrixResult;

void write_results_to_json(const char *filename, MatrixResult *results, int num_matrices);

#endif