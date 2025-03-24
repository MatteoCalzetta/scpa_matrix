#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
    Kernel2
        Ogni warp è responsabile della computazione di una riga della matrice:
            -Uso di shared memory (memoria condivisa)
                -> garantisce coalescenza nell'accesso
                -> permette riduzione attraverso dati nella shared memory

*/


__global__ void spmv_csr_warps_shmem(int M, const int *IRP, const int *JA, 
                                     const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp prende una riga
    int lane = threadIdx.x;  // ID Thread = Indice del thread nel warp
    extern __shared__ double shared_sum[];  // Memoria condivisa per la riduzione

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Ogni thread nel warp processa elementi della riga
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
        }

        // Scrittura risultato parziale all'interno della memoria condivisa
        shared_sum[lane + threadIdx.y * WARP_SIZE] = sum;
        __syncthreads();

        // Riduzione coalescente con memoria condivisa
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                shared_sum[lane + threadIdx.y * WARP_SIZE] += shared_sum[lane + threadIdx.y * WARP_SIZE + offset];
            }
            __syncthreads();
        }

        // Primo thread del warp unico a scrivere il risultato
        if (lane == 0) {
            y[row] = shared_sum[threadIdx.y * WARP_SIZE];  
        }
    }
}



