#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/csr_matrix.h"

// Funzione per il prodotto matrice-vettore in CSR
double *csr_matrtimesvect(CSRMatrix *csr, int *x) {
    double *y = (double *)calloc(csr->M, sizeof(double));
    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            y[i] += csr->AS[j] * x[csr->JA[j]];
        }
    }
    return y;
}

void generate_random_vector(int *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (rand() % 5) + 1; 
    }
}

int main() {
    const char *filename = "test_matrix.mtx";  //questo è un file messo da me per testing, vanno scaricati dal sito e inseriti in una cartella qui

    // Inizializza il generatore di numeri casuali
    srand(time(NULL));

    // Legge la matrice da file e converte in CSR
    CSRMatrix *csr = read_matrix_market(filename);
    if (!csr) {
        printf("\nErrore nella lettura del file %s\n", filename);
        return 1;
    }

    printf("\nConversione in CSR completata con successo!\n\n");
    print_csr(csr);  // Stampa la matrice convertita in CSR

    // Genera il vettore di input con numeri casuali tra 1 e 5, denso ma è da cambiare
    int *x = (int *)malloc(csr->N * sizeof(int));
    if (!x) {
        printf("Errore di allocazione per il vettore x.\n");
        free_csr(csr);
        return 1;
    }

    generate_random_vector(x, csr->N);

    printf("\nVettore di input generato:\n");
    for (int i = 0; i < csr->N; i++) {
        printf("x[%d] = %d\n", i, x[i]);
    }

    double *y = csr_matrtimesvect(csr, x);

    printf("\nVettore y risultante:\n");
    for (int i = 0; i < csr->M; i++) {
        printf("y[%d] = %.1f\n", i, y[i]);  
    }

    // Libera la memoria allocata
    free(x);
    free(y);
    free_csr(csr);  

    return 0;
}