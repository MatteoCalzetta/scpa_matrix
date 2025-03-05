#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr_matrix.h"
#include "../include/hll_matrix.h"

#define HACK_SIZE 32  // Dimensione degli hack

HLLMatrix *convert_csr_to_hll(CSRMatrix *csr) {
    int M = csr->M, N = csr->N;
    int num_hacks = (M + HACK_SIZE - 1) / HACK_SIZE;  // Calcola il numero di hack

    // Alloca la struttura HLL
    HLLMatrix *hll = (HLLMatrix *)malloc(sizeof(HLLMatrix));
    hll->M = M;
    hll->N = N;
    hll->num_hacks = num_hacks;
    hll->hackOffsets = (int *)malloc((num_hacks + 1) * sizeof(int));

    // Conta il numero totale di elementi e crea AS e JA
    int total_nnz = 0;
    for (int h = 0; h < num_hacks; h++) {
        int start_row = h * HACK_SIZE;
        int end_row = (start_row + HACK_SIZE < M) ? start_row + HACK_SIZE : M;

        int maxNR = 0;
        for (int i = start_row; i < end_row; i++) {
            int nnz_row = csr->IRP[i + 1] - csr->IRP[i];
            if (nnz_row > maxNR) maxNR = nnz_row;
        }
        total_nnz += (end_row - start_row) * maxNR;  // Spazio per questo hack
    }

    hll->AS = (double *)malloc(total_nnz * sizeof(double));
    hll->JA = (int *)malloc(total_nnz * sizeof(int));

    // Popola AS e JA con i dati divisi per hack
    int offset = 0;
    for (int h = 0; h < num_hacks; h++) {
        hll->hackOffsets[h] = offset;
        int start_row = h * HACK_SIZE;
        int end_row = (start_row + HACK_SIZE < M) ? start_row + HACK_SIZE : M;

        int maxNR = 0;
        for (int i = start_row; i < end_row; i++) {
            int nnz_row = csr->IRP[i + 1] - csr->IRP[i];
            if (nnz_row > maxNR) maxNR = nnz_row;
        }

        // Scrivi AS e JA con padding se necessario
        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * maxNR;  // Offset dentro l'hack
            int row_nnz = csr->IRP[i + 1] - csr->IRP[i];

            for (int j = 0; j < row_nnz; j++) {
                hll->AS[offset + row_offset + j] = csr->AS[csr->IRP[i] + j];
                hll->JA[offset + row_offset + j] = csr->JA[csr->IRP[i] + j];
            }

            // Padding con 0 se la riga ha meno di maxNR elementi
            for (int j = row_nnz; j < maxNR; j++) {
                hll->AS[offset + row_offset + j] = 0.0;
                hll->JA[offset + row_offset + j] = -1;  // Indice non valido
            }
        }

        offset += (end_row - start_row) * maxNR;
    }

    hll->hackOffsets[num_hacks] = offset;
    return hll;
}

// Funzione per stampare la matrice HLL per debugging
void print_hll_matrix(HLLMatrix *hll) {
    printf("\n=== MATRICE IN FORMATO HLL ===\n");
    printf("Numero di righe: %d, Numero di colonne: %d\n", hll->M, hll->N);
    printf("Numero di hack: %d (hackSize = %d)\n", hll->num_hacks, HACK_SIZE);

    for (int h = 0; h < hll->num_hacks; h++) {
        int start_offset = hll->hackOffsets[h];
        int end_offset = hll->hackOffsets[h + 1];
        int num_rows = (h == hll->num_hacks - 1) ? (hll->M % HACK_SIZE) : HACK_SIZE;

        printf("\n--- Hack %d ---\n", h);
        for (int i = 0; i < num_rows; i++) {
            printf("Riga %d:", h * HACK_SIZE + i);
            for (int j = 0; j < (end_offset - start_offset) / num_rows; j++) {
                int idx = start_offset + i * ((end_offset - start_offset) / num_rows) + j;
                printf(" (%d, %.2f)", hll->JA[idx], hll->AS[idx]);
            }
            printf("\n");
        }
    }
}

// Funzione per liberare la memoria
void free_hll(HLLMatrix *hll) {
    free(hll->hackOffsets);
    free(hll->JA);
    free(hll->AS);
    free(hll);
}