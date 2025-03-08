#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <json-c/json.h>
#include "../include/json_results.h"
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"
#include "../include/openMP_prim.h"
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudacsr.h"
#include "../include/matrix.h"  // Lista delle matrici

#define MATRIX_DIR "../build/test_matrix/"  

void generate_random_vector(double *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (rand() % 5) + 1; 
    }
}

int main() {
    int thread_counts[] = {2, 4, 8, 16, 32, 40};
    srand(time(NULL));

    struct json_object *json_results = json_object_new_array();  // Array JSON per tutti i risultati

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
        
        // Creiamo l'oggetto JSON per questa matrice
        struct json_object *json_matrix = json_object_new_object();
        json_object_object_add(json_matrix, "matrix", json_object_new_string(matrix_filenames[i]));

        // **Esecuzione Serial**
        double execution_time_serial = csr_matrtimesvect(csr, x, y);
        json_object_object_add(json_matrix, "serial_time", json_object_new_double(execution_time_serial));
        printf("[Seriale] Matrice: %s | Tempo: %.10f s\n", matrix_filenames[i], execution_time_serial);

        // **Esecuzione CUDA**
        double execution_time_cuda = csr_matvec_cuda(csr, x, y);
        double flops_cuda = (2.0 * csr->NZ) / execution_time_cuda;
        json_object_object_add(json_matrix, "cuda_time", json_object_new_double(execution_time_cuda));
        json_object_object_add(json_matrix, "cuda_flops", json_object_new_double(flops_cuda));
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], execution_time_cuda, flops_cuda);

        // **Esecuzione OpenMP con diversi thread**
        struct json_object *json_openmp_results = json_object_new_array();
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
            double execution_time_openmp = csr_matvec_openmp(csr, x, y, num_threads, row_partition);
            double exec_flop = 2.0 * csr->NZ / (execution_time_openmp * 1e9);
            printf("[OpenMP] Matrice: %s | Threads: %d | FLOPS: %.10f | Tempo: %.10f s\n", 
                   matrix_filenames[i], num_threads, exec_flop, execution_time_openmp);

            // Creiamo un oggetto JSON per questa configurazione di OpenMP
            struct json_object *json_openmp_entry = json_object_new_object();
            json_object_object_add(json_openmp_entry, "threads", json_object_new_int(num_threads));
            json_object_object_add(json_openmp_entry, "time", json_object_new_double(execution_time_openmp));
            json_object_object_add(json_openmp_entry, "flops", json_object_new_double(exec_flop));

            // Aggiungiamo al JSON OpenMP
            json_object_array_add(json_openmp_results, json_openmp_entry);

            free(row_partition);
        }

        json_object_object_add(json_matrix, "openmp_results", json_openmp_results);
        json_object_array_add(json_results, json_matrix);

        free(x);
        free(y);
        free_csr(csr);
        free_hll(hll);
    }

    // Scriviamo i risultati sul file JSON
    FILE *file = fopen("results.json", "w");
    if (file) {
        fprintf(file, "%s", json_object_to_json_string_ext(json_results, JSON_C_TO_STRING_PRETTY));
        fclose(file);
        printf("Risultati salvati in results.json\n");
    } else {
        printf("Errore: impossibile aprire results.json per la scrittura.\n");
    }

    json_object_put(json_results);  // Cleanup della memoria JSON

    return 0;
}