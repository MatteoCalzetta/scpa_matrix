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
    
    int lane = threadIdx.x;          
    int warpIdInBlock = threadIdx.y; 
    int global_row = blockIdx.x * blockDim.y + warpIdInBlock;
    
    if (global_row >= M) return;
 
    int block_id = global_row / HACK_SIZE;
    int local_row = global_row % HACK_SIZE;
    if (block_id >= d_hll_matrix->num_blocks) return;
 
    const HLLBlock *block = &d_hll_matrix->blocks[block_id];
    int max_nz_per_row = block->max_nz_per_row;
    int row_offset = local_row * max_nz_per_row;
 
    double sum = 0.0;
 
    for (int j = lane; j < max_nz_per_row; j += WARP_SIZE) {
        int idx = row_offset + j;
        int col = block->JA[idx];
        double val = block->AS[idx];
        sum += val * d_x[col];
    }
 
 
    // Riduzione in shared memory 
    extern __shared__ double sdata[];
 
    // Ogni warp ha a disposizione 32 elementi in shared.
    // Indice nel vettore shared per questo thread:
    int s_index = warpIdInBlock * WARP_SIZE + lane;
 
    // Scriviamo la somma parziale in shared
    sdata[s_index] = sum;
    __syncthreads();
 
    // Riduzione logaritmica
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            sdata[s_index] += sdata[s_index + offset];
        }
        __syncthreads();
    }
 
    if (lane == 0) {
        d_y[global_row] = sdata[s_index];
    }
}
 