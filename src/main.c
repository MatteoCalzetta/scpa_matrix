#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "../include/json_results.h"
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"
#include "../include/openMP_prim.h"
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudacsr.h"
#include "../include/matrix.h"  // Inclusione della lista delle matrici

#define MATRIX_DIR "../build/test_matrix/"  

void generate_random_vector(double *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (rand() % 5) + 1; 
    }
}

int main() {
    int thread_counts[] = {2, 4, 8, 16, 32, 40};
    srand(time(NULL));

    for (int i = 0; i < num_matrices; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s%s", MATRIX_DIR, matrix_filenames[i]);

        // Legge la matrice e converte in formato CSR
        CSRMatrix *csr = read_matrix_market(filename);
        if (!csr) {
            printf("Errore: impossibile leggere il file %s\n", filename);
            continue;
        }

        HLLMatrix *hll = convert_csr_to_hll(csr);
        // print_hll_matrix(hll);

        double *x = (double *)malloc(csr->N * sizeof(double));
        if (!x) {
            printf("Errore: allocazione fallita per il vettore x (file: %s)\n", filename);
            free_csr(csr);
            continue;
        }
        generate_random_vector(x, csr->N);

        double *y = (double *)calloc(csr->M, sizeof(double));
        if (!y) {
            printf("Errore: allocazione fallita per il vettore y (file: %s)\n", filename);
            free(x);
            free_csr(csr);
            continue;
        }
        
        double execution_time = csr_matrtimesvect(csr, x, y);
        printf("[Seriale] Matrice: %s | Tempo: %.10f s\n", matrix_filenames[i], execution_time);

        double execution_time_cuda = csr_matvec_cuda(csr, x, y);
        printf("[CUDA] Matrice: %s | Tempo: %.10f s\n", matrix_filenames[i], execution_time_cuda);

        double flops = (2.0 * csr->NZ) / execution_time_cuda;
        printf("[CUDA] Matrice: %s | FLOPS: %.10f\n", matrix_filenames[i], flops);

        for (int j = 0; j < (sizeof(thread_counts)/sizeof(int)); j++) {
            int num_threads = thread_counts[j];

            if (csr->M < num_threads) {
                continue;
            }

            int *row_partition = (int *)malloc(num_threads * sizeof(int));
            if (!row_partition) {
                printf("Errore: allocazione fallita per row_partition (file: %s, threads: %d)\n", filename, num_threads);
                free(x);
                free(y);
                free_csr(csr);
                continue;
            }

            balance_load(csr, num_threads, row_partition);
            execution_time = csr_matvec_openmp(csr, x, y, num_threads, row_partition);
            double exec_flop = 2.0 * csr->NZ / (execution_time * 1e9);
            printf("[OpenMP] Matrice: %s | Threads: %d | FLOPS: %.10f | Tempo: %.10f s\n", 
                   matrix_filenames[i], num_threads, exec_flop, execution_time);

            save_results_to_json("results.json", filename, num_threads, execution_time);

            free(row_partition);
        }

        free(x);
        free(y);
        free_csr(csr);
        free_hll(hll);
    }

    return 0;
}