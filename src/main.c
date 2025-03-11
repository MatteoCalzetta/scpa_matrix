//PER COMPILARE: "cmake .." -> "make -j$(nproc)" nella cartella /build
//PER RICOMPILARE: "rm -rf CMakeCache.txt cmake_install.cmake CMakeFiles/ libcuda_kernels.a progetto.out Makefile"
//                  poi rieseguire comandi PER COMPILARE

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
#include "../include/matrix.h"

#define MATRIX_DIR "../build/test_matrix/"  

void generate_random_vector(double *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (rand() % 5) + 1; 
    }
}

double compute_norm(double *store, double *y, int M) {
    double norm = 0.0;
    for (int i = 0; i < M; i++) {
        double diff = store[i] - y[i];
        norm += diff * diff;
    }
    return sqrt(norm);
}

int main() {
    int thread_counts[] = {2, 4, 8, 16, 32, 40};
    srand(time(NULL));

    MatrixResult results[num_matrices];

    for (int i = 0; i < num_matrices; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s%s", MATRIX_DIR, matrix_filenames[i]);

        // Legge la matrice e converte in CSR
        CSRMatrix *csr = read_matrix_market(filename);
        if (!csr) {
            printf("Errore: impossibile leggere il file %s\n", filename);
            continue;
        }

        HLLMatrix *hll = convert_csr_to_hll(csr);
        
        double *x = (double *)malloc(csr->N * sizeof(double));
        double *y = (double *)calloc(csr->M, sizeof(double));
        double *y2 = (double *)calloc(csr->M, sizeof(double));
        double *store = (double *)calloc(csr->M, sizeof(double));
        if (!x || !y) {
            printf("Errore: allocazione fallita per i vettori x o y (file: %s)\n", filename);
            free_csr(csr);
            continue;
        }
        generate_random_vector(x, csr->N);

        snprintf(results[i].matrix_name, sizeof(results[i].matrix_name), "%s", matrix_filenames[i]);

        // **Esecuzione Serial**
        results[i].serial.time = csr_matrtimesvect(csr, x, y);
        printf("[Seriale] Matrice: %s | Tempo: %.10f s\n", matrix_filenames[i], results[i].serial.time);
        memcpy(store, y, csr->M * sizeof(double));

        // **Esecuzione CUDA**
        results[i].cuda.time = csr_matvec_cuda(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / results[i].cuda.time;
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        //double norm_value = compute_norm(store, y2, csr->M);
        //printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);


        // **Esecuzione OpenMP**
        results[i].num_openmp = 0;
        for (int j = 0; j < (sizeof(thread_counts) / sizeof(int)); j++) {
            if (csr->M < thread_counts[j]) continue;
            
            int *row_partition = (int *)malloc(thread_counts[j] * sizeof(int));
            if (!row_partition) {
                printf("Errore: allocazione fallita per row_partition (file: %s, threads: %d)\n", filename, thread_counts[j]);
                free(x);
                free(y);
                free_csr(csr);
                continue;
            }

            balance_load(csr, thread_counts[j], row_partition);

            double norm_value = compute_norm(store, y, csr->M);
            printf("Norma L2 tra seriale e openmp: %f\n", norm_value);

            results[i].openmp_results[results[i].num_openmp].threads = thread_counts[j];
            results[i].openmp_results[results[i].num_openmp].time = csr_matvec_openmp(csr, x, y, thread_counts[j], row_partition);
            results[i].openmp_results[results[i].num_openmp].flops = 2.0 * csr->NZ / (results[i].openmp_results[results[i].num_openmp].time * 1e9);
            results[i].num_openmp++;
            
            printf("[OpenMP] Matrice: %s | Threads: %d | FLOPS: %.10f | Tempo: %.10f s | NZ: %d\n", 
                   matrix_filenames[i], thread_counts[j], results[i].openmp_results[results[i].num_openmp-1].flops, results[i].openmp_results[results[i].num_openmp-1].time, csr->NZ);
            
            free(row_partition);
        }

        free(x);
        free(y);
        free_csr(csr);
        free_hll(hll);
    }

    // Scriviamo i risultati sul file JSON
    write_results_to_json("results.json", results, num_matrices);
    
    return 0;
}
