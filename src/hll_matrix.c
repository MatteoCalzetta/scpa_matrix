#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/mmio.h"
#include "../include/csr_matrix.h"
#include "../include/hll_matrix.h"

#define HackSize 32

typedef struct {
    int col;
    double val;
}Entry;

// Calcola i non-zeri per riga
void calculate_nz_per_row_csr(const CSRMatrix *csr, int *nz_per_row) {
    for (int i = 0; i < csr->M; i++) {
        nz_per_row[i] = csr->IRP[i + 1] - csr->IRP[i];
    }
}

// Trova il massimo numero di non-zeri tra le righe del blocco
int find_max_nz_csr(const int *nz_per_row, int start_row, int end_row) {
    int max_nz = 0;
    for (int i = start_row; i < end_row; i++) {
        if (nz_per_row[i] > max_nz) max_nz = nz_per_row[i];
    }
    return max_nz;
}

// Conversione da CSR a HLL
HLLMatrix *convert_csr_to_hll(const CSRMatrix *csr) {
    HLLMatrix *hll_matrix = malloc(sizeof(HLLMatrix));
    hll_matrix->M = csr->M;
    hll_matrix->N = csr->N;
    hll_matrix->num_blocks = (csr->M + HackSize - 1) / HackSize;
    hll_matrix->blocks = malloc(hll_matrix->num_blocks * sizeof(HLLBlock));

    int *nz_per_row = calloc(csr->M, sizeof(int));
    calculate_nz_per_row_csr(csr, nz_per_row);

    // Scorri blocchi
    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > csr->M) end_row = csr->M;

        hll_matrix->blocks[block_idx].max_nz_per_row = find_max_nz_csr(nz_per_row, start_row, end_row);
        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;

        hll_matrix->blocks[block_idx].JA = calloc(size_of_arrays, sizeof(int));
        hll_matrix->blocks[block_idx].AS = calloc(size_of_arrays, sizeof(double));

        memset(hll_matrix->blocks[block_idx].JA, -1, size_of_arrays * sizeof(int));
        memset(hll_matrix->blocks[block_idx].AS, 0, size_of_arrays * sizeof(double));

        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = csr->IRP[i];
            int row_nz_end = csr->IRP[i + 1];

            int pos = 0;
            int last_col_idx = -1;

            for (int j = row_nz_start; j < row_nz_end; j++) {
                if (pos >= max_nz_per_row) break;
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = csr->JA[j];
                hll_matrix->blocks[block_idx].AS[index] = csr->AS[j];
                last_col_idx = csr->JA[j];
                pos++;
            }

            // Padding
            while (pos < max_nz_per_row) {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }
    }

    free(nz_per_row);
    return hll_matrix;
}

// Stampa matrice HLL
void print_hll_matrix_csr(const HLLMatrix *hll_matrix) {
    printf("\n=== MATRICE IN FORMATO HLL (CSR) ===\n");
    printf("Numero di righe: %d, Numero di colonne: %d\n", hll_matrix->M, hll_matrix->N);
    printf("Numero di blocchi: %d (HackSize = %d)\n", hll_matrix->num_blocks, HackSize);

    for (int b = 0; b < hll_matrix->num_blocks; b++) {
        printf("\n--- Blocco %d ---\n", b);
        int rows_in_block = (b == hll_matrix->num_blocks - 1) ? (hll_matrix->M % HackSize) : HackSize;
        int max_nz_per_row = hll_matrix->blocks[b].max_nz_per_row;

        for (int i = 0; i < rows_in_block; i++) {
            printf("Riga %d:", b * HackSize + i);
            for (int j = 0; j < max_nz_per_row; j++) {
                int idx = i * max_nz_per_row + j;
                if (hll_matrix->blocks[b].JA[idx] != -1) {
                    printf(" (%d, %.2f)", hll_matrix->blocks[b].JA[idx], hll_matrix->blocks[b].AS[idx]);
                }
            }
            printf("\n");
        }
    }
}

void free_hll_matrix_col(HLLMatrix *hll) {
    if (!hll) return;

    for (int b = 0; b < hll->num_blocks; b++) {
        HLLBlock *block = &hll->blocks[b];
        free(block->JA);
        free(block->AS);
    }

    free(hll->blocks);
    free(hll);
}

// Free HLL
void free_hll_matrix(HLLMatrix *hll_matrix) {
    if (!hll_matrix) return;
    for (int i = 0; i < hll_matrix->num_blocks; i++) {
        free(hll_matrix->blocks[i].JA);
        free(hll_matrix->blocks[i].AS);
    }
    free(hll_matrix->blocks);
    free(hll_matrix);
}


HLLMatrix *convert_csr_to_hll_column_major(const CSRMatrix *csr) {
    int M = csr->M;
    int N = csr->N;

    HLLMatrix *hll = malloc(sizeof(HLLMatrix));
    hll->M = M;
    hll->N = N;
    hll->num_blocks = (M + HackSize - 1) / HackSize;
    hll->blocks = malloc(hll->num_blocks * sizeof(HLLBlock));

    int *nz_per_row = calloc(M, sizeof(int));
    for (int i = 0; i < M; i++) {
        nz_per_row[i] = csr->IRP[i + 1] - csr->IRP[i];
    }

    for (int b = 0; b < hll->num_blocks; b++) {
        int start = b * HackSize;
        int end = (b + 1) * HackSize;
        if (end > M) end = M;
        int rows = end - start;

        int max_nz = 0;
        for (int i = start; i < end; i++) {
            if (nz_per_row[i] > max_nz) max_nz = nz_per_row[i];
        }

        int size = rows * max_nz;
        int *JA = malloc(size * sizeof(int));
        double *AS = malloc(size * sizeof(double));
        for (int i = 0; i < size; i++) {
            JA[i] = -1;
            AS[i] = 0.0;
        }

        for (int i = start; i < end; i++) {
            int local_row = i - start;
            int pos = 0;
            int row_start = csr->IRP[i];
            int row_end = csr->IRP[i + 1];

            for (int j = row_start; j < row_end && pos < max_nz; j++, pos++) {
                int idx = pos * rows + local_row; // column-major index
                JA[idx] = csr->JA[j];
                AS[idx] = csr->AS[j];
            }
        }

        hll->blocks[b].JA = JA;
        hll->blocks[b].AS = AS;
        hll->blocks[b].max_nz_per_row = max_nz;
        hll->blocks[b].rows_in_block = rows;
    }

    free(nz_per_row);
    return hll;
}