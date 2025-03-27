#define WARP_SIZE 32
#define HACK_SIZE 32

__global__ void matvec_hll_column_kernel(const HLLMatrix *d_hll, const double *d_x, double *d_y, int M) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= M) return;

    int block_id = global_row / HACK_SIZE;
    int local_row = global_row % HACK_SIZE;

    const HLLBlock *block = &d_hll->blocks[block_id];
    int rows = block->rows_in_block;
    int max_nz = block->max_nz_per_row;

    double sum = 0.0;
    for (int c = 0; c < max_nz; ++c) {
        int idx = c * rows + local_row; // accesso column-major
        int col = block->JA[idx];
        double val = block->AS[idx];
        if (col >= 0) sum += val * d_x[col];
    }

    d_y[global_row] = sum;
}