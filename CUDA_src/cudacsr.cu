#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"

#define WARP_SIZE 32  

__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void csr_matvec_kernel(int M, double *AS, int *JA, int *IRP, double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (row < M) {
        double sum = 0.0;

        for (int j = IRP[row] + threadIdx.x; j < IRP[row + 1]; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
        }

        sum = warpReduceSum(sum);

        if (threadIdx.x == 0) {
            y[row] = sum;
        }
    }
}

double csr_matvec_cuda(CSRMatrix *csr, double *x, double *y) {

    printf("Eseguendo CUDA\n");

    int *d_JA, *d_IRP;
    double *d_AS, *d_x, *d_y;

    int M = csr->M;
    int NZ = csr->NZ;

    // Allocazione memoria Pinned (Page-Locked) per x e y
    cudaHostAlloc((void **)&x, csr->N * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **)&y, M * sizeof(double), cudaHostAllocDefault);

    // Allocazione memoria Device
    cudaMalloc((void **)&d_AS, NZ * sizeof(double));
    cudaMalloc((void **)&d_JA, NZ * sizeof(int));
    cudaMalloc((void **)&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc((void **)&d_x, csr->N * sizeof(double));
    cudaMalloc((void **)&d_y, M * sizeof(double));

    // Creazione eventi CUDA per misurare il tempo
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il cronometro prima dell'esecuzione del kernel
    cudaEventRecord(start);

    // Copia dati in modo asincrono
    cudaMemcpyAsync(d_AS, csr->AS, NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_JA, csr->JA, NZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_IRP, csr->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_x, x, csr->N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(WARP_SIZE, 1);
    dim3 gridDim((M + blockDim.y - 1) / blockDim.y);

    csr_matvec_kernel<<<gridDim, blockDim>>>(M, d_AS, d_JA, d_IRP, d_x, d_y);

    // Copia asincrona del risultato dalla GPU alla CPU
    cudaMemcpyAsync(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // Aspetta che la copia asincrona sia completata
    cudaDeviceSynchronize();

    // Ferma il cronometro
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcola il tempo trascorso in millisecondi
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Stampa il tempo in millisecondi
    printf("Tempo di esecuzione: %.10f ms\n", elapsedTime);

    // Free memoria Device
    cudaFree(d_AS);
    cudaFree(d_JA);
    cudaFree(d_IRP);
    cudaFree(d_x);
    cudaFree(d_y);

    // Free memoria Pinned
    cudaFreeHost(x);
    cudaFreeHost(y);

    // Restituisci il tempo in secondi (converte i millisecondi in secondi)
    return elapsedTime / 1000.0;
}
