#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudahll1d.h"


#define HACK_SIZE 32

// Kernel per trasporre la matrice HLL
__global__ void transposeHLLKernel(const int *JA, const double *AS, const int *hackOffsets,
                                    int *JA_T, double *AS_T, int M, int N, int maxWidth) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int start = hackOffsets[row];
    int width = (hackOffsets[row + 1] - start) / HACK_SIZE;

    for (int i = 0; i < width; i++) {
        int col = JA[start + i];
        if (col != -1) {
            int new_pos = col * maxWidth + row;
            AS_T[new_pos] = AS[start + i];
            JA_T[new_pos] = row;
        }
    }
}

// Kernel per il prodotto matrice vettore
__global__ void spmvHLLKernel(const double *AS, const int *JA, const int *hackOffsets,
                               const double *x, double *y, int M, int maxWidth) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    double sum = 0.0;
    int start = hackOffsets[row];
    int width = (hackOffsets[row + 1] - start) / HACK_SIZE;

    for (int i = 0; i < width; i++) {
        int col = JA[start + i];
        if (col != -1) {
            sum += AS[start + i] * x[col];
        }
    }

    y[row] = sum;
}

// Funzione principale per la trasposizione e il prodotto matrice vettore
void spmvHLL_CUDA(HLLMatrix *hll, const double *h_x, double *h_y) {
    int *d_JA, *d_JA_T;
    double *d_AS, *d_AS_T, *d_x, *d_y;
    int *d_hackOffsets;
    int total_nnz = hll->hackOffsets[hll->num_hacks];

    cudaMalloc(&d_JA, total_nnz * sizeof(int));
    cudaMalloc(&d_AS, total_nnz * sizeof(double));
    cudaMalloc(&d_hackOffsets, (hll->num_hacks + 1) * sizeof(int));
    cudaMalloc(&d_x, hll->N * sizeof(double));
    cudaMalloc(&d_y, hll->M * sizeof(double));

    cudaMemcpy(d_JA, hll->JA, total_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, hll->AS, total_nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hackOffsets, hll->hackOffsets, (hll->num_hacks + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, hll->N * sizeof(double), cudaMemcpyHostToDevice);

    // Calcola la trasposta
    cudaMalloc(&d_JA_T, total_nnz * sizeof(int));
    cudaMalloc(&d_AS_T, total_nnz * sizeof(double));

    int blockSize = 256;
    int gridSize = (hll->M + blockSize - 1) / blockSize;
    transposeHLLKernel<<<gridSize, blockSize>>>(d_JA, d_AS, d_hackOffsets, d_JA_T, d_AS_T, hll->M, hll->N, HACK_SIZE);
    cudaDeviceSynchronize();

    // Esegue il prodotto matrice-vettore
    spmvHLLKernel<<<gridSize, blockSize>>>(d_AS_T, d_JA_T, d_hackOffsets, d_x, d_y, hll->M, HACK_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, hll->M * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera la memoria
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_hackOffsets);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_JA_T);
    cudaFree(d_AS_T);
}