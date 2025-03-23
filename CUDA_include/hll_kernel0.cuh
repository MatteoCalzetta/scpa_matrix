#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudahll.h"
#include "../include/hll_matrix.h"
#include <cuda.h>
#include <iostream>

#define HACK_SIZE 32
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128

__global__ void matvec_Hll_cuda_SH(const HLLMatrix *d_hll_matrix, const double *d_x, double *d_y, int M) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row >= M) return;

    int block_id = global_row / HACK_SIZE;
    int local_row = global_row % HACK_SIZE;

    if (block_id >= d_hll_matrix->num_blocks) return;

    const HLLBlock *block = &d_hll_matrix->blocks[block_id];
    int row_offset = local_row * block->max_nz_per_row;

    double sum = 0.0;

    for (int j = 0; j < block->max_nz_per_row; j++) {
        int col_idx = block->JA[row_offset + j];
        double value = block->AS[row_offset + j];
        sum += value * d_x[col_idx];
    }

    d_y[global_row] = sum;
}