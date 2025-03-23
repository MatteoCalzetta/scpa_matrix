#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr_matrix.h"

#define HackSize 32

// Struttura per memorizzare i blocchi HLL
typedef struct {
    int *JA;   // Indici delle colonne
    double *AS; // Valori
    int max_nz_per_row;  // Numero massimo di non nulli per riga
} HLLBlock;

// Struttura per memorizzare l'intera matrice HLL
typedef struct {
    int M, N;
    int num_blocks;
    HLLBlock *blocks;  // Array di blocchi HLL
} HLLMatrix;

// Funzioni di conversione e utility
void calculate_nz_per_row_csr(const CSRMatrix *csr, int *nz_per_row);
int find_max_nz_csr(const int *nz_per_row, int start_row, int end_row);
HLLMatrix *convert_csr_to_hll(const CSRMatrix *csr);
void print_hll_matrix_csr(const HLLMatrix *hll_matrix);
void free_hll_matrix(HLLMatrix *hll_matrix);
void convert_block_to_column_major(HLLBlock *block, int rows_in_block);
void hll_matvec_openmp(HLLMatrix *hll_matrix, double *x, double *y, int num_threads);

#endif // HLL_MATRIX_H
