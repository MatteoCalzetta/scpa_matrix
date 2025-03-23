#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudahll.h"
#include "../include/hll_matrix.h"
#include <cuda.h>
#include <iostream>

#define WARP_SIZE 32
#define HACK_SIZE 32

// Kernel: ogni warp elabora una riga della matrice HLL.
// Si assume che la matrice HLL sia organizzata in blocchi da HACK_SIZE righe
// e che per ogni riga siano memorizzati 'max_nz_per_row' elementi.
__global__ void matvec_Hll_cuda_warp(const HLLMatrix *d_hll_matrix, 
                                     const double *d_x, 
                                     double *d_y, 
                                     int M) {
    // In una configurazione 2D dove blockDim.x = 32 (WARP_SIZE)
    // e blockDim.y = number of warps per blocco,
    // la hardware linearizza i thread in modo che le righe (threadIdx.y) 
    // rimangano costanti all'interno di ogni warp.
    int lane = threadIdx.x;       // indice della lane nel warp (0..31)
    int warpIdInBlock = threadIdx.y;  // ogni warp è identificato dalla sua riga nel blocco
    int global_row = blockIdx.x * blockDim.y + warpIdInBlock; // ogni warp processa una riga

    // Se la riga globale supera M, esci
    if (global_row >= M)
        return;

    // Determina a quale blocco HLL appartiene la riga e l'indice locale
    int block_id = global_row / HACK_SIZE;
    int local_row = global_row % HACK_SIZE;
    if (block_id >= d_hll_matrix->num_blocks)
        return;

    const HLLBlock *block = &d_hll_matrix->blocks[block_id];
    int max_nz_per_row = block->max_nz_per_row;
    int row_offset = local_row * max_nz_per_row;

    double sum = 0.0;
    // Ogni lane processa elementi della riga con stride WARP_SIZE
    for (int j = lane; j < max_nz_per_row; j += WARP_SIZE) {
        int idx = row_offset + j;
        int col = block->JA[idx];
        double val = block->AS[idx];
        sum += val * d_x[col];
    }

    // Riduzione a livello di warp usando __shfl_down_sync
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Il thread di lane 0 scrive il risultato
    if (lane == 0)
        d_y[global_row] = sum;
}

