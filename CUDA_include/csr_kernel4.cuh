#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
    Kernel4
        Ogni warp è responsabile della computazione di una riga della matrice:
            -Uso di CacheL2: dato viene cercato in cache prima di accedere alla memoria globale
                -> di base non garantisce coalescenza nell'accesso
                -> permette una riduzione della latenza rispetto alla memoria globale
            -Uso di riduzione per sommare i risultati attraverso l'uso di __shfl_sync
                -> uso collaborativo tra cacheL2 e __shfl_sync
                -> rischio di errore accumulato per ogni thread rispetto al formato double

*/


__global__ void spmv_csr_warps_cachel2(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // ID Thread = Indice del thread nel warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread somma i propri elementi della riga con uso di cacheL2
        for (int j = row_start + lane; j < row_end; j += 32) {
            int col = __ldg(&JA[j]);  // Carica indice colonna da cacheL2
            sum += __ldg(&AS[j]) * __ldg(&x[col]);  // Carica valori da cache
        }

        // Riduzione parallela con __shfl_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_sync(0xFFFFFFFF, sum, lane + offset);
        }


        if (lane == 0) {
            y[row] = sum;
        }
    }
}
