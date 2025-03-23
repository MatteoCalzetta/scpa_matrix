#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudahll.h"
#include "../include/hll_matrix.h"
#include <cuda.h>
#include <iostream>

#define HACK_SIZE 32
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128

// Kernel per il prodotto matrice-vettore usando shared memory
// su una matrice HLL in formato column-major.
__global__ void matvec_Hll_cuda_shared(const HLLMatrix *d_hll_matrix, 
    const double *d_x, 
    double *d_y, 
    int M) {
// Ogni blocco CUDA elabora un blocco HLL.
    int block_id = blockIdx.x;
    if (block_id >= d_hll_matrix->num_blocks) return;

// Determina il numero di righe nel blocco:
    int rows_in_block = HACK_SIZE;
    if (block_id == d_hll_matrix->num_blocks - 1) {
        int rem = M % HACK_SIZE;
        if (rem != 0) rows_in_block = rem;
    }   

// La configurazione 2D del blocco:
//   blockDim.x = rows_in_block (un thread per riga)
//   blockDim.y = T, dove T è il numero di thread per riga per la riduzione (ad esempio, 32)
    int local_row = threadIdx.x;   // indice della riga all'interno del blocco HLL
    int thread_col = threadIdx.y;  // indice per la riduzione (colonna del tile)
    int global_row = block_id * HACK_SIZE + local_row;
    if (global_row >= M) return;

    const HLLBlock *block = &d_hll_matrix->blocks[block_id];
    int max_nz_per_row = block->max_nz_per_row;

// Allocazione dinamica della shared memory:
// La shared memory è organizzata come una matrice di dimensione (rows_in_block x blockDim.y)
    extern __shared__ double shared_sum[];
    int sm_idx = local_row * blockDim.y + thread_col;
    shared_sum[sm_idx] = 0.0;
    __syncthreads();

// Ogni thread carica il prodotto se il suo thread_col è minore di max_nz_per_row.
    if (thread_col < max_nz_per_row) {
// Dati in formato column-major: l'elemento della riga local_row e colonna thread_col è:
//    index = thread_col * rows_in_block + local_row
        int index = thread_col * rows_in_block + local_row;
        int col = block->JA[index];
        double val = block->AS[index];
        shared_sum[sm_idx] = val * d_x[col];
    }
    __syncthreads();

// Riduzione lungo la dimensione y per ogni riga.
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (thread_col < stride) {
            shared_sum[sm_idx] += shared_sum[local_row * blockDim.y + thread_col + stride];
        }
    __syncthreads();
    }

// Il thread con thread_col == 0 scrive il risultato finale per la riga globale.
    if (thread_col == 0) {
        d_y[global_row] = shared_sum[local_row * blockDim.y];
    }
}
