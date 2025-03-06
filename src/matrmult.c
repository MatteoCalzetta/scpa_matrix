#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"


/*
// Funzione per il prodotto matrice-vettore BASE in CSR
double csr_matrtimesvect(CSRMatrix *csr, int *x, double *y) {
    struct timespec start, end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            y[i] += csr->AS[j] * x[csr->JA[j]];
        }
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}
*/


// Funzione per il prodotto matrice-vettore BASE in CSR
double csr_matrtimesvect(CSRMatrix *csr, int *x, double *y) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Avvia il timer

    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            y[i] += csr->AS[j] * x[csr->JA[j]];
        }
    }

    end = clock();  // Ferma il timer

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calcola il tempo in secondi

    printf("Fattooooo \n");

    return cpu_time_used;
}
