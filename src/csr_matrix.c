#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "../include/csr_matrix.h"

// Funzione per leggere una matrice MatrixMarket e convertirla direttamente in CSR
CSRMatrix* read_matrix_market(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Errore nell'apertura del file %s\n", filename);
        return NULL;
    }

    // **Ignora i commenti e leggi l'intestazione**
    char header[256];
    bool is_symmetric = false, is_pattern = false;

    while (fgets(header, sizeof(header), file)) {
        if (header[0] == '%') {
            if (strstr(header, "symmetric")) is_symmetric = true;
            if (strstr(header, "pattern")) is_pattern = true;
        } else {
            break; // La riga successiva contiene le dimensioni
        }
    }

    // **Legge dimensioni della matrice**
    int M, N, NZ;
    sscanf(header, "%d %d %d", &M, &N, &NZ);

    // Se la matrice è simmetrica, dobbiamo contare gli elementi duplicati
    int total_nz = is_symmetric ? (2 * NZ) : NZ;

    // **Allocazione degli array temporanei**
    int *row_indices = (int *)malloc(total_nz * sizeof(int));
    int *col_indices = (int *)malloc(total_nz * sizeof(int));
    double *values = (double *)malloc(total_nz * sizeof(double));

    if (!row_indices || !col_indices || !values) {
        printf("Errore di allocazione memoria.\n");
        goto cleanup;
    }

    // **Legge gli elementi della matrice**
    int count = 0;
    for (int i = 0; i < NZ; i++) {
        int r, c;
        double v = 1.0;  // Default se "pattern"

        if (is_pattern) {
            fscanf(file, "%d %d", &r, &c);
        } else {
            fscanf(file, "%d %d %lf", &r, &c, &v);
        }

        r--; c--; // Converti a indice 0-based
        row_indices[count] = r;
        col_indices[count] = c;
        values[count++] = v;

        // **Se la matrice è simmetrica e non è sulla diagonale, aggiungiamo il valore speculare**
        if (is_symmetric && r != c) {
            row_indices[count] = c;
            col_indices[count] = r;
            values[count++] = v;
        }
    }
    fclose(file);

    // **Ora costruiamo il formato CSR**
    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!csr) {
        printf("Errore di allocazione della struttura CSR.\n");
        goto cleanup;
    }

    csr->M = M;
    csr->N = N;
    csr->NZ = count; // Aggiornato con il nuovo valore effettivo
    csr->AS = (double *)malloc(csr->NZ * sizeof(double));
    csr->JA = (int *)malloc(csr->NZ * sizeof(int));
    csr->IRP = (int *)calloc((M + 1), sizeof(int));

    if (!csr->AS || !csr->JA || !csr->IRP) {
        printf("Errore di allocazione degli array CSR.\n");
        goto cleanup;
    }

    // **Conta gli elementi per riga per costruire IRP**
    for (int i = 0; i < csr->NZ; i++)
        csr->IRP[row_indices[i] + 1]++;

    // **Calcola i puntatori di riga (somma cumulativa)**
    for (int i = 1; i <= M; i++)
        csr->IRP[i] += csr->IRP[i - 1];

    // **Riempie AS e JA**
    for (int i = 0; i < csr->NZ; i++) {
        int row = row_indices[i];
        int pos = csr->IRP[row]++;
        csr->JA[pos] = col_indices[i];
        csr->AS[pos] = values[i];
    }

    // **Ripristina IRP**
    for (int i = M; i > 0; i--)
        csr->IRP[i] = csr->IRP[i - 1];
    csr->IRP[0] = 0;

    // **Pulizia memoria temporanea**
    free(row_indices);
    free(col_indices);
    free(values);

    return csr;

cleanup:
    free(row_indices);
    free(col_indices);
    free(values);
    free_csr(csr);
    return NULL;
}

// **Funzione per stampare la matrice in formato CSR**
void print_csr(CSRMatrix *csr) {
    if (!csr) {
        printf("Matrice CSR non valida.\n");
        return;
    }

    printf("Matrice CSR (M = %d, N = %d, NZ = %d)\n", csr->M, csr->N, csr->NZ);

    printf("IRP: ");
    for (int i = 0; i <= csr->M; i++) {
        printf("%d ", csr->IRP[i]);
    }
    printf("\n");

    printf("JA: ");
    for (int i = 0; i < csr->NZ; i++) {
        printf("%d ", csr->JA[i]);
    }
    printf("\n");

    printf("AS: ");
    for (int i = 0; i < csr->NZ; i++) {
        printf("%.2f ", csr->AS[i]);
    }
    printf("\n");
}

// **Funzione per liberare la memoria della struttura CSR**
void free_csr(CSRMatrix *csr) {
    if (!csr) return;
    free(csr->AS);
    free(csr->JA);
    free(csr->IRP);
    free(csr);
}