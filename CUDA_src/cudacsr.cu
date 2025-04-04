#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include "../CUDA_include/csr_kernel0.cuh"
#include "../CUDA_include/csr_kernel1.cuh"
#include "../CUDA_include/csr_kernel2.cuh"
#include "../CUDA_include/csr_kernel3.cuh"
#include "../CUDA_include/csr_kernel4.cuh"
#include "../CUDA_include/csr_kernel5.cuh"

#define WARP_SIZE 32

//k5
double spmv_csr_warps_texture(CSRMatrix *csr, const double *x, double *y) {
    int M = csr->M;
    int NZ = csr->NZ;

    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori memoria GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_y;
    float *d_x;  // Texture usa float invece di double

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));
    cudaMalloc(&d_x, csr->N * sizeof(float));  // Texture usa float

    // Convertire double in float prima di copiarlo
    float *h_x = (float*)malloc(csr->N * sizeof(float));
    for (int i = 0; i < csr->N; i++) h_x[i] = (float)x[i];

    // Dati da CPU a GPU
    cudaMemcpy(d_IRP, csr->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, csr->JA, NZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, csr->AS, NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, csr->N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_x);

    // Creazione texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_x;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = csr->N * sizeof(float);

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex_x;
    cudaCreateTextureObject(&tex_x, &resDesc, &texDesc, nullptr);

    // Configurazione blocchi e griglia kernel
    int warp_size = 32;
    int num_warps_per_block = 8;
    dim3 block_dim(warp_size, num_warps_per_block);
    dim3 grid_dim((M + num_warps_per_block - 1) / num_warps_per_block);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il cronometro
    cudaEventRecord(start, stream);

    spmv_csr_warp_texture<<<grid_dim, block_dim, 0, stream>>>(M, d_IRP, d_JA, d_AS, tex_x, d_y);

    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copia il risultato dalla GPU alla CPU
    cudaMemcpyAsync(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    // Distruggere il texture object
    cudaDestroyTextureObject(tex_x);

    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaStreamDestroy(stream);

    return elapsedTime / 1000;  // Restituisce il tempo in secondi
}


//K4
double spmv_csr_warps_cachel2(CSRMatrix *csr, double *x, double *y) {
    int M = csr->M;
    int NZ = csr->NZ;

    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori per la memoria sulla GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, csr->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    // Dati da CPU a GPU
    cudaMemcpyAsync(d_IRP, csr->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_JA, csr->JA, NZ * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_AS, csr->AS, NZ * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, x, csr->N * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemset(d_y, 0, M * sizeof(double));  // Inizializza d_y a zero

    // Creazione degli eventi CUDA per il timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configurazione blocchi e griglia kernel
    int warps_per_block = 4;
    dim3 block_dim(32, warps_per_block); 
    dim3 grid_dim((M + warps_per_block - 1) / warps_per_block);

    // Avvia cronometro
    cudaEventRecord(start, stream);

    spmv_csr_warps_cachel2<<<grid_dim, block_dim, 0, stream>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copia il risultato dalla GPU alla CPU
    cudaMemcpyAsync(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distrugge lo stream CUDA
    cudaStreamDestroy(stream);

    return elapsedTime / 1000;  // Restituisce il tempo in secondi
}





//kernel3
double spmv_csr_warps_shmem_ridpar(CSRMatrix *csr, double *x, double *y) {
    int M = csr->M;
    int NZ = csr->NZ;

    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori per la memoria sulla GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, csr->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    // Dati da CPU a GPU
    cudaMemcpyAsync(d_IRP, csr->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_JA, csr->JA, NZ * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_AS, csr->AS, NZ * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, x, csr->N * sizeof(double), cudaMemcpyHostToDevice, stream);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configurazione blocchi e griglia kernel
    int warps_per_block = 4;
    dim3 block_dim(32, warps_per_block);
    dim3 grid_dim((M + warps_per_block - 1) / warps_per_block);

    size_t shared_mem_size = warps_per_block * 32 * sizeof(double);  // Allocazione della shared memory

    // Avvia cronometro
    cudaEventRecord(start, stream);


    spmv_csr_warps_shmem_ridpar<<<grid_dim, block_dim, shared_mem_size>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Risultato da GPU a CPU
    cudaMemcpyAsync(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distrugge lo stream CUDA
    cudaStreamDestroy(stream);

    return elapsedTime / 1000;  // Restituisce il tempo in secondi
}




//kernel2
double spmv_csr_warps_shmem(CSRMatrix *csr, double *x, double *y) {
    int M = csr->M;
    int NZ = csr->NZ;

    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori per la memoria sulla GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, csr->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    // Copia dati da CPU a GPU
    cudaMemcpyAsync(d_IRP, csr->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_JA, csr->JA, NZ * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_AS, csr->AS, NZ * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, x, csr->N * sizeof(double), cudaMemcpyHostToDevice, stream);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configurazione blocchi e griglia kernel
    int warps_per_block = 4;
    dim3 block_dim(32, warps_per_block);
    dim3 grid_dim((M + warps_per_block - 1) / warps_per_block);

    // Allocazione memoria condivisa
    size_t shared_mem_size = warps_per_block * WARP_SIZE * sizeof(double);

    // Avvia cronometro
    cudaEventRecord(start, stream);

    spmv_csr_warps_shmem<<<grid_dim, block_dim, shared_mem_size>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Risultato da GPU a CPU
    cudaMemcpyAsync(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distrugge lo stream CUDA
    cudaStreamDestroy(stream);

    return elapsedTime / 1000;  // Restituisce il tempo in secondi
}


//kernel1
double spmv_csr_warps(CSRMatrix *h_mat, double *h_x, double *h_y) {
    int M = h_mat->M;
    int NZ = h_mat->NZ;

    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori per la memoria sulla GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, h_mat->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    
    // Copia dati da CPU a GPU
    cudaMemcpy(d_IRP, h_mat->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, h_mat->JA, NZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, h_mat->AS, NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, h_mat->N * sizeof(double), cudaMemcpyHostToDevice);
    

    // Configurazione blocchi e griglia kernel
    dim3 blockDim(WARP_SIZE, 8);
    dim3 gridDim((M + blockDim.y - 1) / blockDim.y);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvio cronometro
    cudaEventRecord(start, stream);

    spmv_csr_warps<<<gridDim, blockDim, 0, stream>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Risultato da GPU a CPU
    cudaMemcpyAsync(h_y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distrugge lo stream CUDA
    cudaStreamDestroy(stream);

    return elapsedTime / 1000; // Converte ms in secondi
}




//kernel0
double spmv_csr_threads(CSRMatrix *h_mat, double *h_x, double *h_y) {
    int M = h_mat->M;
    int NZ = h_mat->NZ;

    // Creazione variabili per GPU
    cudaEvent_t start, stop;
    float elapsedTime;
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    //Allocazione variabili e spostamento da host a gpu
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, h_mat->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    cudaMemcpyAsync(d_IRP, h_mat->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_JA, h_mat->JA, NZ * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_AS, h_mat->AS, NZ * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, h_x, h_mat->N * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Configurazione blocchi e griglia kernel
    int threads_per_block = 256;
    int num_blocks = (M + threads_per_block - 1) / threads_per_block;

    //Cronometro
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    spmv_csr_threads<<<num_blocks, threads_per_block, 0, stream>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cudaMemcpyAsync(h_y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    //Free variabili GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaStreamDestroy(stream);

    return elapsedTime / 1000; //elapsedTime è nativamente in ms
}