#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr_matrix.h"


// Funzione per leggere una matrice MatrixMarket e convertirla in CSR
CSRMatrix* read_matrix_market(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error during the opening of %s\n", filename);
        return NULL;
    }

    // Ignora l'intestazione e i commenti usando fgetc()
    char ch;
    do {
        ch = fgetc(file); // Legge un carattere alla volta
        if (ch == '%') {
            while (fgetc(file) != '\n'); // Salta l'intera riga
        }
    } while (ch == '%');
    ungetc(ch, file);

    // Legge dimensioni della matrice
    int M, N, NZ;
    if (fscanf(file, "%d %d %d", &M, &N, &NZ) != 3) {
        printf("Error reading matrix dimensions.\n");
        fclose(file);
        return NULL;
    }

    // Allocazione degli array temporanei
    int *row_indices = (int *)malloc(NZ * sizeof(int));
    int *col_indices = (int *)malloc(NZ * sizeof(int));
    double *values = (double *)malloc(NZ * sizeof(double));

    if (!row_indices || !col_indices || !values) {
        printf("Memory allocation failed.\n");
        fclose(file);
        free(row_indices);
        free(col_indices);
        free(values);
        return NULL;
    }

    // Legge gli elementi della matrice
    for (int i = 0; i < NZ; i++) {
        if (fscanf(file, "%d %d %lf", &row_indices[i], &col_indices[i], &values[i]) != 3) {
            printf("Error reading matrix elements.\n");
            fclose(file);
            free(row_indices);
            free(col_indices);
            free(values);
            return NULL;
        }
        row_indices[i]--; // Converti a 0-based index
        col_indices[i]--;
    }
    fclose(file);

    // Allocazione struttura CSR
    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!csr) {
        printf("Memory allocation for CSRMatrix failed.\n");
        free(row_indices);
        free(col_indices);
        free(values);
        return NULL;
    }

    csr->M = M;
    csr->N = N;
    csr->NZ = NZ;
    csr->AS = (double *)malloc(NZ * sizeof(double));
    csr->JA = (int *)malloc(NZ * sizeof(int));
    csr->IRP = (int *)calloc((M + 1), sizeof(int));

    if (!csr->AS || !csr->JA || !csr->IRP) {
        printf("Memory allocation for CSR arrays failed.\n");
        free_csr(csr);
        free(row_indices);
        free(col_indices);
        free(values);
        return NULL;
    }

    // Conta gli elementi per riga per costruire IRP
    for (int i = 0; i < NZ; i++)
        csr->IRP[row_indices[i] + 1]++;

    // Calcola i puntatori di riga (somma cumulativa)
    for (int i = 1; i <= M; i++)
        csr->IRP[i] += csr->IRP[i - 1];

    // Riempie AS e JA
    int *row_offset = (int *)malloc(M * sizeof(int));
    if (!row_offset) {
        printf("Memory allocation for row_offset failed.\n");
        free_csr(csr);
        free(row_indices);
        free(col_indices);
        free(values);
        return NULL;
    }
    memcpy(row_offset, csr->IRP, M * sizeof(int));

    for (int i = 0; i < NZ; i++) {
        int row = row_indices[i];
        int pos = row_offset[row]++;
        csr->JA[pos] = col_indices[i];
        csr->AS[pos] = values[i];
    }

    // Pulizia memoria temporanea
    free(row_indices);
    free(col_indices);
    free(values);
    free(row_offset);

    return csr;
}

// Funzione per stampare la matrice in formato CSR
void print_csr(CSRMatrix *csr) {
    printf("Matrice CSR (M = %d, N = %d, NZ = %d)\n", csr->M, csr->N, csr->NZ);

    printf("IRP: ");
    for (int i = 0; i <= csr->M; i++) printf("%d ", csr->IRP[i]);
    printf("\n");

    printf("JA: ");
    for (int i = 0; i < csr->NZ; i++) printf("%d ", csr->JA[i]);
    printf("\n");

    printf("AS: ");
    for (int i = 0; i < csr->NZ; i++) printf("%.1f ", csr->AS[i]);
    printf("\n");
}

// Funzione per liberare la memoria della struttura CSR
void free_csr(CSRMatrix *csr) {
    if (csr) {
        free(csr->AS);
        free(csr->JA);
        free(csr->IRP);
        free(csr);
    }
}