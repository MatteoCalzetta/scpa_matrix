#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudahll.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "../include/matrix.h"
#include "../CUDA_include/hll_kernel0.cuh"
#include "../CUDA_include/hll_kernel2.cuh"
#include "../CUDA_include/hll_kernel3.cuh"



#define HACK_SIZE 32
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 128

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

matrixPerformance parallel_hll_cuda_v2(HLLMatrix *hllMatrixHost, double *x_h, double *y_h) {
    double *d_y;
    double *d_x;
    int M = hllMatrixHost->M;

    cudaDeviceReset();
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // NOTA: y_h è passato dal main (deve essere allocato lì con dimensione M)

    // Allocazione della struttura HLL sulla GPU
    HLLMatrix *d_hll_matrix;
    cudaMalloc(&d_hll_matrix, sizeof(HLLMatrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    HLLBlock *d_blocks;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(HLLBlock));
    cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(HLLBlock *), cudaMemcpyHostToDevice);

    // Trasferimento di ciascun blocco sulla GPU
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        HLLBlock *block = &hllMatrixHost->blocks[i];

        int *d_JA;
        double *d_AS;
        int rows_in_block = (i == hllMatrixHost->num_blocks - 1) ?
                            (hllMatrixHost->M % HACK_SIZE) : HACK_SIZE;
        if (rows_in_block == 0)
            rows_in_block = HACK_SIZE;

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

    // Timer per misurare le prestazioni
    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);

    // Trasferimento del vettore x sulla GPU
    cudaMalloc((void **)&d_x, hllMatrixHost->N * sizeof(double));
    cudaMemcpy(d_x, x_h, hllMatrixHost->N * sizeof(double), cudaMemcpyHostToDevice);

    // Alloca il vettore y sulla GPU (il risultato verrà copiato in y_h passato dal main)
    cudaMalloc((void **)&d_y, M * sizeof(double));
    cudaMemset(d_y, 0, M * sizeof(double));

    // Configurazione dinamica del kernel:
    // Se M >= 1024, usiamo 32 warps per blocco (32x32=1024 thread per blocco)
    // Altrimenti, calcoliamo il numero necessario per coprire M righe.
    int warps_per_block;
    if (M >= 1024) {
        warps_per_block = 32;
    } else {
        warps_per_block = (M + WARP_SIZE - 1) / WARP_SIZE;
        if (warps_per_block < 1)
            warps_per_block = 1;
    }
    // Configura il blocco 2D: blockDim.x = WARP_SIZE (32 thread per warp),
    // blockDim.y = warps_per_block (numero di righe per blocco)
    dim3 blockDim(WARP_SIZE, warps_per_block);
    // Ogni blocco elabora warps_per_block righe, quindi il numero di blocchi necessari è:
    int grid_x = (M + warps_per_block - 1) / warps_per_block;
    dim3 gridDim(grid_x);

    // Lancia il kernel
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    printf("Prima di chiamare il kernel\n");
    matvec_Hll_cuda_warp<<<gridDim, blockDim>>>(d_hll_matrix, d_x, d_y, M);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    printf("Dopo aver chiamato il kernel\n");
    sdkStopTimer(&timer);

    matrixPerformance node = {0};
    node.seconds = sdkGetTimerValue(&timer) / 1000.0f; // ms -> s

    // Copia il risultato dal device all'area y_h passata dal main
    cudaMemcpy(y_h, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // Pulizia: libera la memoria allocata per i blocchi trasferiti
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
    
    // Ora y_h contiene il vettore risultato, gestito dal main
    return node;
}



//kernel 0
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



matrixPerformance parallel_hll_cuda_v3(HLLMatrix *hllMatrixHost, double *x_h, double *y_h) {
    double *d_y;
    double *d_x;
    int M = hllMatrixHost->M;
    // (Opzionale) Reset del device
    cudaDeviceReset();
    // Per debug: controlla la memoria disponibile
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // NOTA: y_h è allocato dal main (dimensione M)
    // 1) Allocazione della struttura HLL sulla GPU
    HLLMatrix *d_hll_matrix;
    cudaMalloc(&d_hll_matrix, sizeof(HLLMatrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLLMatrix), cudaMemcpyHostToDevice);
    // 2) Allocazione e copia dei blocchi
    HLLBlock *d_blocks;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(HLLBlock));
    cudaMemcpy(&d_hll_matrix->blocks, &d_blocks, sizeof(HLLBlock *), cudaMemcpyHostToDevice);
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        HLLBlock *block = &hllMatrixHost->blocks[i];
        int rows_in_block = (i == hllMatrixHost->num_blocks - 1) ?
                            (hllMatrixHost->M % HACK_SIZE) : HACK_SIZE;
        if (rows_in_block == 0) {
            rows_in_block = HACK_SIZE;
        }
        // Dimensioni array JA e AS del blocco
        size_t JA_size = block->max_nz_per_row * rows_in_block * sizeof(int);
        size_t AS_size = block->max_nz_per_row * rows_in_block * sizeof(double);
        // Alloca e copia device
        int *d_JA;
        double *d_AS;
        cudaMalloc(&d_JA, JA_size);
        cudaMemcpy(d_JA, block->JA, JA_size, cudaMemcpyHostToDevice);
        cudaMalloc(&d_AS, AS_size);
        cudaMemcpy(d_AS, block->AS, AS_size, cudaMemcpyHostToDevice);
        // Prepara un HLLBlock "device" temporaneo per scriverlo nell’array d_blocks
        HLLBlock d_block = *block;  // copia di tutti i campi
        d_block.JA = d_JA;          // ma con i puntatori device
        d_block.AS = d_AS;
        cudaMemcpy(&d_blocks[i], &d_block, sizeof(HLLBlock), cudaMemcpyHostToDevice);
    }
    // 3) Timer per misurare le prestazioni
    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);
    // 4) Copia del vettore x sulla GPU
    cudaMalloc((void **)&d_x, hllMatrixHost->N * sizeof(double));
    cudaMemcpy(d_x, x_h, hllMatrixHost->N * sizeof(double), cudaMemcpyHostToDevice);
    // 5) Alloca il vettore y sulla GPU
    cudaMalloc((void **)&d_y, M * sizeof(double));
    cudaMemset(d_y, 0, M * sizeof(double));
    // 6) Configurazione del kernel
    // Calcolo del numero di warps per blocco
    int warps_per_block;
    if (M >= 1024) {
        warps_per_block = 32;  // 32 warps => 32*32=1024 threads
    } else {
        warps_per_block = (M + WARP_SIZE - 1) / WARP_SIZE;
        if (warps_per_block < 1) {
            warps_per_block = 1;
        }
    }
    // blockDim.x = 32 (un warp ha 32 lane), blockDim.y = numero di warps per blocco
    dim3 blockDim(WARP_SIZE, warps_per_block);
    // Ogni blocco processa warps_per_block righe; quante righe totali? M.
    int grid_x = (M + warps_per_block - 1) / warps_per_block;
    dim3 gridDim(grid_x);
    // *Allocazione della shared memory*:
    // Per ogni warp servono 32 * sizeof(double), e abbiamo "warps_per_block" warps.
    size_t sharedSize = warps_per_block * WARP_SIZE * sizeof(double);
    // 7) Lancio del kernel con sharedSize
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    printf("Prima di chiamare il kernel (shared memory)...\n");
    matvec_Hll_cuda_warp_shared<<<gridDim, blockDim, sharedSize>>>(
        d_hll_matrix, d_x, d_y, M
    );
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    printf("Dopo aver chiamato il kernel (shared memory)\n");
    sdkStopTimer(&timer);
    matrixPerformance node = {0};
    node.seconds = sdkGetTimerValue(&timer) / 1000.0f; // ms -> s
    // 8) Copia il risultato dal device all'area y_h passata dal main
    cudaMemcpy(y_h, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
    // 9) Pulizia: libera la memoria allocata per i blocchi trasferiti
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
    // y_h contiene il risultato
    return node;
}


matrixPerformance parallel_hll_column_cuda(const HLLMatrix *hll, const double *x_h, double *y_h) {
    int M = hll->M, N = hll->N;

    // Allocazione device
    double *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));
    cudaMemcpy(d_x, x_h, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(double));

    // Copia struttura HLL
    HLLMatrix *d_hll;
    cudaMalloc(&d_hll, sizeof(HLLMatrix));
    cudaMemcpy(d_hll, hll, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    HLLBlock *d_blocks;
    cudaMalloc(&d_blocks, hll->num_blocks * sizeof(HLLBlock));
    cudaMemcpy(&d_hll->blocks, &d_blocks, sizeof(HLLBlock *), cudaMemcpyHostToDevice);

    // Copia blocchi
    for (int i = 0; i < hll->num_blocks; i++) {
        HLLBlock tmp = hll->blocks[i];
        int size = tmp.rows_in_block * tmp.max_nz_per_row;

        cudaMalloc(&tmp.JA, size * sizeof(int));
        cudaMemcpy(tmp.JA, hll->blocks[i].JA, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&tmp.AS, size * sizeof(double));
        cudaMemcpy(tmp.AS, hll->blocks[i].AS, size * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(&d_blocks[i], &tmp, sizeof(HLLBlock), cudaMemcpyHostToDevice);
    }

    // Configura kernel
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    
    int threads_per_block = 128;
    int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;
    
    // allinea ai multipli di sm_count
    if (blocks_per_grid % sm_count != 0)
        blocks_per_grid = ((blocks_per_grid / sm_count) + 1) * sm_count;
    
    dim3 blockDim(threads_per_block);
    dim3 gridDim(blocks_per_grid);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    matvec_hll_column_kernel<<<gridDim, blockDim>>>(d_hll, d_x, d_y, M);
    cudaDeviceSynchronize();
    
    sdkStopTimer(&timer);

    cudaMemcpy(y_h, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    for (int i = 0; i < hll->num_blocks; i++) {
        cudaFree(hll->blocks[i].JA);
        cudaFree(hll->blocks[i].AS);
    }
    cudaFree(d_blocks);
    cudaFree(d_hll);

    matrixPerformance result = {0};
    result.seconds = sdkGetTimerValue(&timer) / 1000.0f;
    sdkDeleteTimer(&timer);
    return result;
}
