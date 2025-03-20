#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
    Kernel4
        Ogni warp è responsabile della computazione di una riga della matrice:
            -Uso di CacheL2
                -> di base non garantisce coalescenza nell'accesso
                -> permette una riduzione della latenza rispetto alla memoria globale
            -Uso di riduzione per sommare i risultati attraverso l'uso di __shfl_sync
                -> rischio di errore accumulato per ogni thread rispetto al formato double

*/



//cacheL2 - __shfl_sync_
__global__ void spmv_csr_warps_shmem_ridpar2(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // Indice del thread nel warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread somma i propri elementi della riga
        for (int j = row_start + lane; j < row_end; j += 32) {
            int col = __ldg(&JA[j]);  // Carica indice colonna con cache L2
            sum += __ldg(&AS[j]) * __ldg(&x[col]);  // Carica valori con cache
        }

        // Riduzione parallela con __shfl_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_sync(0xFFFFFFFF, sum, lane + offset);
        }

        // Solo il primo thread del warp scrive il risultato
        if (lane == 0) {
            y[row] = sum;
        }
    }
}



/*
//uso della cacheL2 - __shfl_down_sync
__global__ void spmv_csr_warps_shmem_ridpar2(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // Indice del thread nel warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread somma i propri elementi della riga
        for (int j = row_start + lane; j < row_end; j += 32) {
            int col = __ldg(&JA[j]);  // Usa __ldg per migliorare cache locality
            sum += __ldg(&AS[j]) * __ldg(&x[col]);  // Carica i dati in cache
        }

        // Riduzione parallela più efficiente con __shfl_down_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Solo il primo thread del warp scrive il risultato
        if (lane == 0) {
            y[row] = sum;  // Evita atomicAdd se ogni warp gestisce una riga
        }
    }
}
*/



/*
__global__ void spmv_csr_warps_shmem_ridpar2(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // Indice del thread nel warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread somma i propri elementi della riga
        for (int j = row_start + lane; j < row_end; j += 32) {
            int col = JA[j];  // Evitiamo letture duplicate dalla memoria globale
            sum += AS[j] * x[col];
        }

        // Riduzione parallela con __shfl_xor_sync (senza shared memory)
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }

        // Scrittura sicura in memoria globale
        if (lane == 0) {
            atomicAdd(&y[row], sum);  // Usa atomicAdd se ci sono più warps per riga
        }
    }
}
*/

