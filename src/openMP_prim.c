#include <stdio.h>
#include <stdlib.h>
#include "../include/csr_matrix.h"
#include "../include/openMP_prim.h"

void balance_load(CSRMatrix *csr, int num_threads, int *thr_partition) {
    int *row_nnz = (int *)malloc(csr->M * sizeof(int));  
    int *nnz_per_thread_count = (int *)calloc(num_threads, sizeof(int));  // Conta gli nnz assegnati per thread
    double total_nnz = 0.0;
    
    // 1️⃣ Conta i nonzeri per riga e calcola il totale
    for (int i = 0; i < csr->M; i++) {
        row_nnz[i] = csr->IRP[i + 1] - csr->IRP[i];
        total_nnz += row_nnz[i];
    }

    double target_workload = total_nnz / num_threads;  
    double current_workload = 0.0;
    int thread_id = 0;
    thr_partition[0] = 0;

    // 2️⃣ Suddivisione migliorata
    for (int i = 0; i < csr->M; i++) {
        current_workload += row_nnz[i];
        nnz_per_thread_count[thread_id] += row_nnz[i];  

        // Se il thread ha accumulato abbastanza lavoro e ci sono ancora thread disponibili
        if (current_workload >= target_workload && thread_id < num_threads - 1) {
            thr_partition[thread_id + 1] = i + 1;  // Il prossimo thread partirà da qui
            thread_id++;
            current_workload = 0.0;
        }
    }

    // 3️⃣ Ultimo thread prende le righe rimanenti
    thr_partition[num_threads] = csr->M;  

   

    free(row_nnz);
    free(nnz_per_thread_count);
}

double csr_matvec_openmp(CSRMatrix *csr, int *x, double *y, int num_threads, int *row_partition) {
    double start_time = omp_get_wtime(); // Misura il tempo totale

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start_row = row_partition[thread_id];
        int end_row = (thread_id == num_threads - 1) ? csr->M : row_partition[thread_id + 1];

        double local_start_time = omp_get_wtime(); // Inizio del calcolo effettivo

        for (int i = start_row; i < end_row; i++) {
            double sum = 0.0;
            for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
                sum += csr->AS[j] * x[csr->JA[j]];
            }
            y[i] = sum;
        }

        double local_end_time = omp_get_wtime(); // Fine del calcolo effettivo

        //printf("Thread %d - Tempo di calcolo effettivo: %f secondi\n", thread_id, local_end_time - local_start_time);
    }

    double end_time = omp_get_wtime();
    return end_time - start_time; // Ritorna il tempo totale
}