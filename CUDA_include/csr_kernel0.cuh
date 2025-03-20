#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
    Kernel0
        Ogni thread è responsabile della computazione di una riga della matrice:
            -Uso di memoria globale -> coalescenza non garantita

*/


__global__ void spmv_csr_threads(int M, const int *IRP, const int *JA, 
                                const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];
        for (int j = row_start; j < row_end; j++) {
            //y[row] += AS[j] * x[JA[j]];
            sum += AS[j] * x[JA[j]];
        }
        y[row] = sum;
    }
}
