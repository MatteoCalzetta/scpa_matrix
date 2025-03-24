#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
    Kernel3
        Ogni warp è responsabile della computazione di una riga della matrice:
            -Uso di shared memory (memoria condivisa)
                -> garantisce coalescenza nell'accesso
            -Uso di riduzione per sommare i risultati attraverso l'uso di __shfl_sync
                -> uso collaborativo tra memoria condivisia e __shfl_sync
                -> rischio di errore accumulato per ogni thread rispetto al formato double

*/

__global__ void spmv_csr_warps_shmem_ridpar(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // Indice del thread nel warp
    int warp_id = threadIdx.y;  // ID WARP = Indice del warp nel blocco

    __shared__ double shared_sum[32][32];  // Shared memory per riduzione per ogni warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread somma i propri elementi della riga
        for (int j = row_start + lane; j < row_end; j += 32) {
            sum += AS[j] * x[JA[j]];
        }

        // Memorizzazione somma parziale nella shared memory
        shared_sum[warp_id][lane] = sum;
        __syncthreads();


        sum = shared_sum[warp_id][lane];

        // Riduzione con __shfl_sync attraverso lettura coalescente su memoria condivisa
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_sync(0xFFFFFFFF, sum, lane + offset);
        }

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

