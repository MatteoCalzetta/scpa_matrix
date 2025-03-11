/*#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

// Kernel CUDA per prodotto matrice-vettore in formato CSR
__global__ void spmv_csr_kernel(int M, const int *IRP, const int *JA, 
                                const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += AS[j] * x[JA[j]];
        }
        y[row] = sum;
    }
}

// Funzione per eseguire SpMV in CUDA con trasferimenti asincroni
double csr_matvec_cuda(CSRMatrix *h_mat, double *h_x, double *h_y) {
    int M = h_mat->M;
    int NZ = h_mat->NZ;

    // Creazione eventi CUDA per misurare il tempo
    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori per la memoria sulla GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    // Creazione di uno stream CUDA per l'asincronia
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, h_mat->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    // Copia dati dalla CPU alla GPU in modo asincrono
    cudaMemcpyAsync(d_IRP, h_mat->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_JA, h_mat->JA, NZ * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_AS, h_mat->AS, NZ * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, h_x, h_mat->N * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Configurazione griglia CUDA
    int threads_per_block = 256;
    int num_blocks = (M + threads_per_block - 1) / threads_per_block;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il cronometro prima dell'esecuzione del kernel
    cudaEventRecord(start, stream);

    // Lancio del kernel CUDA con lo stesso stream
    spmv_csr_kernel<<<num_blocks, threads_per_block, 0, stream>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    // Ferma il cronometro
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    // Calcola il tempo trascorso in millisecondi
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Stampa il tempo in millisecondi
    printf("Tempo di esecuzione: %.10f ms\n", elapsedTime);

    // Copia il risultato dalla GPU alla CPU in modo asincrono
    cudaMemcpyAsync(h_y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    // Sincronizza lo stream prima di liberare memoria
    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distrugge lo stream CUDA
    cudaStreamDestroy(stream);

    return elapsedTime / 1000;  // Converte ms in secondi
}
    */
