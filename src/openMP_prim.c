#include <stdio.h>
#include <stdlib.h>
#include "../include/csr_matrix.h"
#include "../include/hll_matrix.h"
#include "../include/openMP_prim.h"


int* balance_load(CSRMatrix *csr, int num_threads, int *actual_threads) {
    int *thr_partition = malloc((num_threads + 1) * sizeof(int));
    if (!thr_partition) {
        *actual_threads = 0;
        return NULL;
    }

    int *row_nnz = malloc(csr->M * sizeof(int));
    if (!row_nnz) {
        free(thr_partition);
        *actual_threads = 0;
        return NULL;
    }

    double total_nnz = 0.0;
    for (int i = 0; i < csr->M; i++) {
        row_nnz[i] = csr->IRP[i + 1] - csr->IRP[i];
        total_nnz += row_nnz[i];
    }

    double target_workload = total_nnz / num_threads;
    double current_workload = 0.0;

    int thread_id = 0;
    thr_partition[0] = 0;  // La prima partizione parte dalla riga 0

    for (int i = 0; i < csr->M; i++) {
        current_workload += row_nnz[i];
        if (current_workload >= target_workload && thread_id < num_threads - 1) {
            thread_id++;
            thr_partition[thread_id] = i + 1; 
            current_workload = 0.0;
        }
    }

    thread_id++;
    thr_partition[thread_id] = csr->M;

    // thread_id = numero di partizioni create; e’ anche il numero di thread usati
    *actual_threads = thread_id;

    if (*actual_threads < num_threads) {
        int needed_size = *actual_threads + 1; // +1 per l’ultima entry
        int *temp = realloc(thr_partition, needed_size * sizeof(int));
        if (temp != NULL) {
            thr_partition = temp;
        }
    }

    free(row_nnz);

    // Restituisce il puntatore
    return thr_partition;
}


void csr_matvec_openmp(CSRMatrix *csr, double *x, double *y, int num_threads, int *row_partition) {
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int start_row = row_partition[thread_id];
        int end_row = row_partition[thread_id + 1];

        // Calcolo del prodotto matrice-vettore per il blocco assegnato
        for (int i = start_row; i < end_row; i++) {
            double sum = 0.0;
            for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
                y[i] += csr->AS[j] * x[csr->JA[j]];
            }
        }
    }
}



//prodotto SpMV con direttiva openMP guided
/*
void csr_matvec_openmp(CSRMatrix *csr, double *x, double *y, int num_threads, int *row_partition) {
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    
    for (int i = 0; i < csr->M; i++) {
        double sum = 0.0;
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            sum += csr->AS[j] * x[csr->JA[j]];
        }
        y[i] = sum;
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Total threads used: %d\n", omp_get_num_threads());
        }
    }
}*/




void hll_matvec_openmp(HLLMatrix *hll_matrix, double *x, double *y, int num_threads) {

    #pragma omp parallel for schedule(guided) num_threads(num_threads)

    for (int blockID = 0; blockID < hll_matrix->num_blocks; blockID++) {
        int start_row = blockID * HackSize;
        int end_row = (blockID + 1) * HackSize;
        if (end_row > hll_matrix->M) end_row = hll_matrix->M;  

        int max_nz_per_row = hll_matrix->blocks[blockID].max_nz_per_row;
        int row_offset = 0;  

        for (int i = start_row; i < end_row; i++) {
            double sum = 0.0;

            for (int j = 0; j < max_nz_per_row; j++) {
                int idx = row_offset + j;
                if (hll_matrix->blocks[blockID].JA[idx] != -1) { // Salta il padding
                    sum += hll_matrix->blocks[blockID].AS[idx] * x[hll_matrix->blocks[blockID].JA[idx]];
                }
            }
            y[i] = sum;

            // Aggiorniamo l'offset per la prossima riga
            row_offset += max_nz_per_row;
        }
    }

}