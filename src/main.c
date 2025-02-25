#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h> 
#include "../include/json_results.h"
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"
#include "../include/openMP_prim.h"

#define MATRIX_DIR "test_matrix/"  

void generate_random_vector(int *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (rand() % 5) + 1; 
    }
}

int main() {
    int thread_counts[] = {2, 4, 8, 16, 32, 40};

    struct dirent *entry;
    DIR *dir = opendir(MATRIX_DIR);

    if (!dir) {
        printf("Errore: impossibile aprire la cartella %s\n", MATRIX_DIR);
        return 1;
    }

    srand(time(NULL));

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char filename[256];
        snprintf(filename, sizeof(filename), "%s%s", MATRIX_DIR, entry->d_name);

        CSRMatrix *csr = read_matrix_market(filename);
        if (!csr) {
            printf("%s - Errore nella lettura del file\n", filename);
            continue;
        }

        int *x = (int *)malloc(csr->N * sizeof(int));
        if (!x) {
            printf("%s - Errore di allocazione per il vettore x\n", filename);
            free_csr(csr);
            continue;
        }
        generate_random_vector(x, csr->N);

        double *y = (double *)calloc(csr->M, sizeof(double));
        if (!y) {
            printf("%s - Errore di allocazione per il vettore y\n", filename);
            free(x);
            free_csr(csr);
            continue;
        }

        double execution_time = csr_matrtimesvect(csr, x, y);
        printf("%s - Tempo di esecuzione: %f secondi\n", filename, execution_time);

        for (int i = 0; i < (sizeof(thread_counts)/sizeof(int)); i++) {
            int num_threads = thread_counts[i];

            if (csr->M < num_threads) {
                continue;
            }

            int *row_partition = (int *)malloc(num_threads * sizeof(int));
            if (!row_partition) {
                printf("%s - Errore di allocazione per row_partition\n", filename);
                free(x);
                free(y);
                free_csr(csr);
                continue;
            }

            balance_load(csr, num_threads, row_partition);
            execution_time = csr_matvec_openmp(csr, x, y, num_threads, row_partition);

            save_results_to_json("results.json", filename, num_threads, execution_time);

            free(row_partition);
        }

        free(x);
        free(y);
        free_csr(csr);
    }

    closedir(dir);
    return 0;
}