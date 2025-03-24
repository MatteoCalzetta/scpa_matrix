#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
    Kernel5
        Ogni warp è responsabile della computazione di una riga della matrice:
            -Uso di Texture Memory: memoria separata dalla globale sempre con funzioni di caching
                                    separata da CacheL2
                -> di base non garantisce coalescenza nell'accesso
                -> permette una riduzione della latenza rispetto alla memoria globale
                -> nativamente ottimizzata per accessi non coalescenti
            -Uso di riduzione per sommare i risultati attraverso l'uso di __shfl_sync
                -> uso collaborativo tra Texture Memory e __shfl_sync
                -> rischio di errore accumulato per ogni thread rispetto al formato double

*/

__global__ void spmv_csr_warp_texture(int M, const int *IRP, const int *JA, const double *AS, cudaTextureObject_t tex_x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // ID Thread = Indice del thread nel warp

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        for (int j = row_start + lane; j < row_end; j += 32) {
            float x_val = tex1Dfetch<float>(tex_x, JA[j]);  // Lettura dalla texture
            sum += AS[j] * (double)x_val;  // Convertiamo float -> double
        }

        // Riduzione parallela all'interno del warp con __shfl_sync
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane == 0) {
            y[row] = sum;
        }
    }
}


