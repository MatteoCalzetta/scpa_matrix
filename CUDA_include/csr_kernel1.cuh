#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

//TODO FARE KERNEL SEPARATO PER __shfl_sync, di base ho maggiore prestazione con quella non down

/*
    Kernel1
        Ogni warp è responsabile della computazione di una riga della matrice:
            -Uso di memoria globale 
                -> coalescenza non garantita
            -Uso di riduzione per sommare i risultati di tutti i thread per warp
                -> rischio di errore accumulato per ogni thread rispetto al formato double
                -> uso alternativo di __shfl_down_sync che introduce rischio maggiore di errore a
                    causa della riduzione a cascata

*/

#define WARP_SIZE 32
/*
__global__ void spmv_csr_warps(int M, const int *IRP, const int *JA, 
                              const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y; // Ogni warp lavora su una riga
    int lane = threadIdx.x;  // Indice del thread all'interno del warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread nel warp processa alcuni elementi della riga
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
        }

        // Riduzione in serie: somma i contributi di tutti i thread nel warp
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Il primo thread del warp scrive il risultato nella memoria globale
        if (lane == 0) {
            y[row] = sum;
        }
    }
}
*/

__global__ void spmv_csr_warps(int M, const int *IRP, const int *JA, 
                              const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y; // Ogni warp lavora su una riga
    int lane = threadIdx.x;  // Indice del thread all'interno del warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread nel warp processa alcuni elementi della riga
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
        }

        // Riduzione in serie usando __shfl_sync per sommare i contributi di tutti i thread nel warp
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_sync(0xFFFFFFFF, sum, lane + offset);
        }

        // Il primo thread del warp scrive il risultato nella memoria globale
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

