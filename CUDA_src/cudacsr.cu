#include <cuda_runtime.h>
#include <stdio.h>
#include "../CUDA_include/cudacsr.h"
#include "../include/csr_matrix.h"
#include <cuda.h>
#include <iostream>

/*
__global__ void spmv_csr_kernel(int M, int N, const int *IRP, const int *JA, 
                                    const double *AS, const double *x, double *y) {
    // Indici 2D per organizzare i thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Indice della riga
    int col = threadIdx.y;  // Indice della colonna per ciascun thread

    if (row < M) {
        // Variabile per accumulare il risultato del prodotto per la riga
        double sum = 0.0;
        
        // Otteniamo l'inizio e la fine della riga nella matrice CSR
        int row_start = IRP[row];
        int row_end = IRP[row + 1];
        
        // Verifica se la colonna che questo thread deve calcolare è valida
        if (col < (row_end - row_start)) {
            int index = row_start + col;  // Indice dell'elemento non zero nella riga
            
            // Calcoliamo il prodotto matrice-vettore
            sum += AS[index] * x[JA[index]];
        }
        
        // Somma parziale tra i thread del blocco (riduzione intra-blocco)
        __shared__ double shared_sum[32];  // Un array condiviso per la riduzione
        shared_sum[threadIdx.x] = sum;
        
        // Sincronizzazione tra i thread del blocco
        __syncthreads();
        
        // Riduzione in shared memory per ottenere la somma finale per la riga
        if (threadIdx.x < 16) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 16];
        }
        if (threadIdx.x < 8) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 8];
        }
        if (threadIdx.x < 4) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 4];
        }
        if (threadIdx.x < 2) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 2];
        }
        if (threadIdx.x < 1) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 1];
        }
        
        // Solo il thread 0 di ogni blocco scrive il risultato nella memoria globale
        if (threadIdx.x == 0) {
            atomicAdd(&y[row], shared_sum[0]);
        }
    }
}
*/    



/*
__global__ void spmv_csr_kernel(int M, const int *IRP, const int *JA, 
                                      const double *AS, const double *x, double *y) {
    // Indice della riga assegnata a questo warp
    int row = blockIdx.x; // Ogni blocco gestisce una riga
    
    if (row < M) {
        // Otteniamo l'inizio e la fine della riga nella matrice CSR
        int row_start = IRP[row];
        int row_end = IRP[row + 1];
        
        // Numero di elementi non zero nella riga
        int num_elements = row_end - row_start;
        
        // Ogni thread del warp eseguirà un calcolo per un elemento della riga
        // Inizializza la somma parziale
        double sum = 0.0;

        // Calcoliamo un elemento della riga per thread
        // I thread sono distribuiti tra gli elementi della riga
        for (int j = row_start + threadIdx.x; j < row_end; j += 32) {
            sum += AS[j] * x[JA[j]];  // Prodotto matrice-vettore
        }

        // Riduzione tra i thread del warp (32 thread)
        // Eseguiamo una riduzione in parallelo all'interno del warp
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Solo il thread 0 del warp aggiorna il risultato
        if (threadIdx.x == 0) {
            atomicAdd(&y[row], sum);
        }
    }
}
*/


// Kernel CUDA per prodotto matrice-vettore in formato CSR
__global__ void spmv_csr_kernel(int M, const int *IRP, const int *JA, 
                                const double *AS, const double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];
        for (int j = row_start; j < row_end; j++) {
            //y[row] += AS[j] * x[JA[j]];
            sum += AS[j] * x[JA[j]];
        }
        y[row] = sum;
    }
}


// Funzione per eseguire SpMV in CUDA con trasferimenti asincroni
double csr_matvec_cuda(CSRMatrix *h_mat, double *h_x, double *h_y) {
    int M = h_mat->M;
    int NZ = h_mat->NZ;

    // Creazione eventi CUDA per misurare il tempo
    cudaEvent_t start, stop;
    float elapsedTime;

    // Puntatori per la memoria sulla GPU
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;

    // Creazione di uno stream CUDA per l'asincronia
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, h_mat->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    // Copia dati dalla CPU alla GPU in modo asincrono
    cudaMemcpyAsync(d_IRP, h_mat->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_JA, h_mat->JA, NZ * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_AS, h_mat->AS, NZ * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, h_x, h_mat->N * sizeof(double), cudaMemcpyHostToDevice, stream);


    //TODO  numero stream, blocchi di thread, warp per riga in base al num i NZ, preporc dinamico per 
    //      per avere num ponderati 

    // Configurazione griglia CUDA
    //int threads_per_block = 128;
    //int num_blocks = (M + threads_per_block - 1) / threads_per_block;

    int threads_per_block = 32;  // Ogni warp ha 32 thread
    int num_blocks = M;  // Un blocco per ogni riga

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Avvia il cronometro prima dell'esecuzione del kernel
    cudaEventRecord(start, stream);

    // Lancio del kernel CUDA con lo stesso stream
    spmv_csr_kernel<<<num_blocks, threads_per_block, 0, stream>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    // Ferma il cronometro
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    // Calcola il tempo trascorso in millisecondi
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Stampa il tempo in millisecondi
    printf("Tempo di esecuzione: %.10f ms\n", elapsedTime);

    // Copia il risultato dalla GPU alla CPU in modo asincrono
    cudaMemcpyAsync(h_y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost, stream);

    // Sincronizza lo stream prima di liberare memoria
    cudaStreamSynchronize(stream);

    // Libera memoria sulla GPU
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

    // Distrugge lo stream CUDA
    cudaStreamDestroy(stream);

    return elapsedTime / 1000;  // Converte ms in secondi
}
