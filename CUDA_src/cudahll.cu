#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudahll.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "../include/matrix.h"


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

void configure_grid_warp(int M, int sm_count, int *blocks, int *threads) {
    *threads = min(THREADS_PER_BLOCK, M);
    *blocks = (M + *threads - 1) / *threads;

    if ((*blocks * *threads) > M) {
        *blocks = (M + *threads - 1) / *threads;
    }

    if (*blocks % sm_count != 0) {
        *blocks = (*blocks / sm_count + 1) * sm_count;
    }
}

matrixPerformance parallel_hll_cuda_v1(HLLMatrix *hllMatrixHost, double *x_h) {
    double *d_y;
    double *d_x;
    int M = hllMatrixHost->M;

    cudaDeviceReset();
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    auto *y_h = static_cast<double *>(malloc(hllMatrixHost->M * sizeof(double)));
    if (y_h == nullptr) {
        exit(EXIT_FAILURE);
    }

    HLLMatrix *d_hll_matrix;
    cudaMalloc(&d_hll_matrix, sizeof(HLLMatrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    HLLBlock *d_blocks;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(HLLBlock));
    cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(HLLBlock *), cudaMemcpyHostToDevice);

    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        HLLBlock *block = &hllMatrixHost->blocks[i];

        int *d_JA;
        double *d_AS;

        int rows_in_block = (i == hllMatrixHost->num_blocks - 1) ?
                            (hllMatrixHost->M % HACK_SIZE) : HACK_SIZE;
        if (rows_in_block == 0) rows_in_block = HACK_SIZE;

        size_t JA_size = block->max_nz_per_row * rows_in_block * sizeof(int);
        size_t AS_size = block->max_nz_per_row * rows_in_block * sizeof(double);

        cudaMalloc(&d_JA, JA_size);
        cudaMemcpy(d_JA, block->JA, JA_size, cudaMemcpyHostToDevice);

        cudaMalloc(&d_AS, AS_size);
        cudaMemcpy(d_AS, block->AS, AS_size, cudaMemcpyHostToDevice);

        HLLBlock d_block = *block;
        d_block.JA = d_JA;
        d_block.AS = d_AS;

        cudaMemcpy(&d_blocks[i], &d_block, sizeof(HLLBlock), cudaMemcpyHostToDevice);
    }

    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    cudaMalloc((void **)&d_x, hllMatrixHost->N * sizeof(double));
    cudaMemcpy(d_x, x_h, hllMatrixHost->N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_y, hllMatrixHost->M * sizeof(double));
    cudaMemset(d_y, 0, hllMatrixHost->M * sizeof(double));

    int blocks_per_grid, threads_per_block;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    configure_grid_warp(hllMatrixHost->M, sm_count, &blocks_per_grid, &threads_per_block);

    dim3 BLOCK_DIM1(threads_per_block);
    dim3 GRID_DIM1(blocks_per_grid);

    timer->start();
    matvec_Hll_cuda_SH<<<GRID_DIM1, BLOCK_DIM1>>>(d_hll_matrix, d_x, d_y, hllMatrixHost->M);

    cudaDeviceSynchronize();
    timer->stop();

    matrixPerformance node{};
    node.seconds = timer->getTime()/1000.0f;

    cudaMemcpy(y_h, d_y, hllMatrixHost->M * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        HLLBlock temp_block;
        cudaMemcpy(&temp_block, &d_blocks[i], sizeof(HLLBlock), cudaMemcpyDeviceToHost);
        cudaFree(temp_block.JA);
        cudaFree(temp_block.AS);
    }

    cudaFree(d_blocks);
    cudaFree(d_hll_matrix);
    cudaFree(d_x);
    cudaFree(d_y);

    delete timer;
    delete[] y_h;

    return node;
}

// Kernel per il prodotto matrice-vettore usando shared memory
// su una matrice HLL in formato column-major.
__global__ void matvec_Hll_cuda_shared(const HLLMatrix *d_hll_matrix, 
                                       const double *d_x, 
                                       double *d_y, 
                                       int M) {
    // Ogni blocco CUDA elabora un blocco HLL.
    int block_id = blockIdx.x;
    if (block_id >= d_hll_matrix->num_blocks)
        return;
    
    // Determina il numero di righe nel blocco:
    int rows_in_block = HACK_SIZE;
    if (block_id == d_hll_matrix->num_blocks - 1) {
        int rem = M % HACK_SIZE;
        if (rem != 0)
            rows_in_block = rem;
    }
    
    // La configurazione 2D del blocco:
    //   blockDim.x = rows_in_block (un thread per riga)
    //   blockDim.y = T, dove T è il numero di thread per riga per la riduzione (ad esempio, 32)
    int local_row = threadIdx.x;   // indice della riga all'interno del blocco HLL
    int thread_col = threadIdx.y;  // indice per la riduzione (colonna del tile)
    int global_row = block_id * HACK_SIZE + local_row;
    if (global_row >= M)
        return;
    
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


matrixPerformance parallel_hll_cuda_shared(HLLMatrix *hllMatrixHost, double *x_h, double *y_h) {
    double *d_y;
    double *d_x;
    int M = hllMatrixHost->M;

    cudaDeviceReset();
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Non allocare y_h qui: il vettore risultato è passato dal main

    // Alloca la struttura HLL sulla GPU
    HLLMatrix *d_hll_matrix;
    cudaMalloc(&d_hll_matrix, sizeof(HLLMatrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    // Alloca e trasferisci i blocchi HLL sulla GPU
    HLLBlock *d_blocks;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(HLLBlock));
    cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(HLLBlock *), cudaMemcpyHostToDevice);

    // Per ogni blocco, trasferisci JA e AS (già in formato column-major) sulla GPU
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        HLLBlock *block = &hllMatrixHost->blocks[i];

        int *d_JA;
        double *d_AS;
        int rows_in_block = (i == hllMatrixHost->num_blocks - 1) ?
                            (hllMatrixHost->M % HACK_SIZE) : HACK_SIZE;
        if (rows_in_block == 0) rows_in_block = HACK_SIZE;

        size_t JA_size = block->max_nz_per_row * rows_in_block * sizeof(int);
        size_t AS_size = block->max_nz_per_row * rows_in_block * sizeof(double);

        cudaMalloc(&d_JA, JA_size);
        cudaMemcpy(d_JA, block->JA, JA_size, cudaMemcpyHostToDevice);

        cudaMalloc(&d_AS, AS_size);
        cudaMemcpy(d_AS, block->AS, AS_size, cudaMemcpyHostToDevice);

        HLLBlock d_block = *block;
        d_block.JA = d_JA;
        d_block.AS = d_AS;

        cudaMemcpy(&d_blocks[i], &d_block, sizeof(HLLBlock), cudaMemcpyHostToDevice);
    }

    // Timer
    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    // Trasferisci il vettore x sulla GPU
    cudaMalloc((void **)&d_x, hllMatrixHost->N * sizeof(double));
    cudaMemcpy(d_x, x_h, hllMatrixHost->N * sizeof(double), cudaMemcpyHostToDevice);

    // Alloca il vettore y sulla GPU (il risultato verrà poi copiato in y_h, passato dal main)
    cudaMalloc((void **)&d_y, M * sizeof(double));
    cudaMemset(d_y, 0, M * sizeof(double));

    // Configurazione del kernel:
    // Per ogni blocco HLL, usiamo una configurazione 2D:
    //  - blockDim.x = HACK_SIZE (numero di righe nel blocco)
    //  - blockDim.y = threads_per_row (es. 32, per la riduzione)
    int threads_per_row = 32;  // Deve essere una potenza di 2 e sufficientemente grande per coprire max_nz_per_row
    dim3 blockDim(HACK_SIZE, threads_per_row);
    int grid_x = hllMatrixHost->num_blocks; // Un blocco CUDA per ogni blocco HLL
    dim3 gridDim(grid_x);

    // Calcola la memoria condivisa richiesta per blocco:
    size_t sharedMemSize = HACK_SIZE * threads_per_row * sizeof(double);

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    // Lancia il kernel che usa la shared memory (vedi kernel matvec_Hll_cuda_shared implementato precedentemente)
    matvec_Hll_cuda_shared<<<gridDim, blockDim, sharedMemSize>>>(d_hll_matrix, d_x, d_y, M);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    matrixPerformance node = {0};
    node.seconds = sdkGetTimerValue(&timer) / 1000.0f; // Converti millisecondi in secondi

    // Copia il vettore risultato dalla GPU all'area y_h passata dal main
    cudaMemcpy(y_h, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // Pulizia: libera la memoria allocata per ogni blocco
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        HLLBlock temp_block;
        cudaMemcpy(&temp_block, &d_blocks[i], sizeof(HLLBlock), cudaMemcpyDeviceToHost);
        cudaFree(temp_block.JA);
        cudaFree(temp_block.AS);
    }

    cudaFree(d_blocks);
    cudaFree(d_hll_matrix);
    cudaFree(d_x);
    cudaFree(d_y);

    sdkDeleteTimer(&timer);

    return node;
}
