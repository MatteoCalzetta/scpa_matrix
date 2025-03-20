#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>



__global__ void spmv_csr_warp_texture(int M, const int *IRP, const int *JA, const double *AS, cudaTextureObject_t tex_x, double *y) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni warp elabora una riga
    int lane = threadIdx.x;  // Indice del thread nel warp (0-31)

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Computazione parallela della somma
        for (int j = row_start + lane; j < row_end; j += 32) {
            float x_val = tex1Dfetch<float>(tex_x, JA[j]);  // Lettura dalla texture
            sum += AS[j] * (double)x_val;  // Convertiamo float -> double
        }

        // Riduzione parallela all'interno del warp con `__shfl_down_sync`
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Solo il primo thread del warp scrive il risultato
        if (lane == 0) {
            y[row] = sum;
        }
    }
}


