#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr_matrix.h"
#include "../include/hll_matrix.h"

#define HACK_SIZE 32  // Dimensione degli hack

HLLMatrix *convert_csr_to_hll(CSRMatrix *csr) {
    int M = csr->M, N = csr->N;
    int num_hacks = (M + HACK_SIZE - 1) / HACK_SIZE;  // Numero totale di hack

    // **Alloca la struttura HLL**
    HLLMatrix *hll = (HLLMatrix *)malloc(sizeof(HLLMatrix));
    hll->M = M;
    hll->N = N;
    hll->num_hacks = num_hacks;
    hll->hackOffsets = (int *)malloc((num_hacks + 1) * sizeof(int));

    // **Calcola il numero totale di elementi**
    int total_nnz = 0;
    int *maxNR_per_hack = (int *)malloc(num_hacks * sizeof(int));  // Array per salvare maxNR di ogni hack

    for (int h = 0; h < num_hacks; h++) {
        int start_row = h * HACK_SIZE;
        int end_row = (start_row + HACK_SIZE < M) ? start_row + HACK_SIZE : M;

        int maxNR = 0;
        for (int i = start_row; i < end_row; i++) {
            int nnz_row = csr->IRP[i + 1] - csr->IRP[i];
            if (nnz_row > maxNR) maxNR = nnz_row;
        }
        maxNR_per_hack[h] = maxNR;
        total_nnz += (end_row - start_row) * maxNR;
    }

    // **Alloca AS e JA**
    hll->AS = (double *)calloc(total_nnz, sizeof(double));
    hll->JA = (int *)malloc(total_nnz * sizeof(int));

    // Inizializza JA a -1 per distinguere i valori di padding
    for (int i = 0; i < total_nnz; i++) {
        hll->JA[i] = -1;
    }

    // **Popola AS e JA**
    int offset = 0;
    for (int h = 0; h < num_hacks; h++) {
        hll->hackOffsets[h] = offset;
        int start_row = h * HACK_SIZE;
        int end_row = (start_row + HACK_SIZE < M) ? start_row + HACK_SIZE : M;
        int maxNR = maxNR_per_hack[h];

        for (int i = start_row; i < end_row; i++) {
            int row_offset = offset + (i - start_row) * maxNR;  // Offset dentro l'hack
            int row_nnz = csr->IRP[i + 1] - csr->IRP[i];

            for (int j = 0; j < row_nnz; j++) {
                hll->AS[row_offset + j] = csr->AS[csr->IRP[i] + j];
                hll->JA[row_offset + j] = csr->JA[csr->IRP[i] + j];
            }

            // Padding con -1 su JA per identificare valori nulli
            for (int j = row_nnz; j < maxNR; j++) {
                hll->JA[row_offset + j] = -1;
            }
        }
        offset += (end_row - start_row) * maxNR;
    }

    hll->hackOffsets[num_hacks] = offset;

    free(maxNR_per_hack);
    return hll;
}

// **Funzione per stampare la matrice HLL per debugging**
void print_hll_matrix(HLLMatrix *hll) {
    printf("\n=== MATRICE IN FORMATO HLL ===\n");
    printf("Numero di righe: %d, Numero di colonne: %d\n", hll->M, hll->N);
    printf("Numero di hack: %d (hackSize = %d)\n", hll->num_hacks, HACK_SIZE);

    for (int h = 0; h < hll->num_hacks; h++) {
        int start_offset = hll->hackOffsets[h];
        int end_offset = hll->hackOffsets[h + 1];
        int num_rows = (h == hll->num_hacks - 1) ? (hll->M % HACK_SIZE) : HACK_SIZE;
        int row_width = (end_offset - start_offset) / num_rows;

        printf("\n--- Hack %d ---\n", h);
        for (int i = 0; i < num_rows; i++) {
            printf("Riga %d:", h * HACK_SIZE + i);
            for (int j = 0; j < row_width; j++) {
                int idx = start_offset + i * row_width + j;
                if (hll->JA[idx] != -1) {
                    printf(" (%d, %.2f)", hll->JA[idx], hll->AS[idx]);
                }
            }
            printf("\n");
        }
    }
}

// **Funzione per liberare la memoria della matrice HLL**
void free_hll(HLLMatrix *hll) {
    free(hll->hackOffsets);
    free(hll->JA);
    free(hll->AS);
    free(hll);
}


