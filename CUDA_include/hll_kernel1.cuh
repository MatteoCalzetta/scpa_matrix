#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudahll.h"
#include "../include/hll_matrix.h"
#include <cuda.h>
#include <iostream>
 
#define WARP_SIZE 32
#define HACK_SIZE 32
 
// Kernel: ogni warp elabora una riga della matrice HLL.
// La riduzione non usa più __shfl_down_sync, bensì shared memory.
__global__ void matvec_Hll_cuda_warp_shared(const HLLMatrix *d_hll_matrix,
                                     const double *d_x,
                                     double *d_y,
                                     int M)
{
    // In una configurazione 2D dove blockDim.x = 32 (WARP_SIZE)
    // e blockDim.y = number of warps per blocco,
    // l'hardware linearizza i thread in modo che le righe (threadIdx.y)
    // rimangano costanti all'interno di ogni warp.
    int lane = threadIdx.x;          // indice della lane nel warp (0..31)
    int warpIdInBlock = threadIdx.y; // ID del warp all'interno del blocco
    int global_row = blockIdx.x * blockDim.y + warpIdInBlock;
    // ognuno di questi warps elabora una riga
 
    // Se la riga globale supera M, esci
    if (global_row >= M) return;
 
    // Determina a quale blocco HLL appartiene la riga e l'indice locale
    int block_id = global_row / HACK_SIZE;
    int local_row = global_row % HACK_SIZE;
    if (block_id >= d_hll_matrix->num_blocks) return;
 
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
 
    // ---------------------------------------------------------------------
    // Riduzione in shared memory
    // ---------------------------------------------------------------------
    // Allocazione dinamica esterna: "extern _shared_ double sdata[];"
    // La dimensione totale è: blockDim.y * WARP_SIZE (double).
    extern __shared__ double sdata[];
 
    // Ogni warp ha a disposizione 32 elementi in shared.
    // Indice nel vettore shared per questo thread:
    int s_index = warpIdInBlock * WARP_SIZE + lane;
 
    // Scriviamo la somma parziale in shared
    sdata[s_index] = sum;
    __syncthreads();
 
    // Riduzione logaritmica classica
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            sdata[s_index] += sdata[s_index + offset];
        }
        __syncthreads();
    }
 
    // Solo la lane 0 del warp scrive il risultato finale
    if (lane == 0) {
        d_y[global_row] = sdata[s_index];
    }
}
 