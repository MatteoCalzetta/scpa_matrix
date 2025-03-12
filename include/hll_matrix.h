#ifndef HLL_MATRIX_H
#define HLL_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr_matrix.h"

#define HACK_SIZE 32  // Dimensione degli hack

typedef struct {
    int M, N;       // Numero di righe e colonne
    int num_hacks;  // Numero di hack
    int *hackOffsets;  // Indici di inizio degli hack
    int *JA;        // Indici di colonna
    double *AS;     // Valori non nulli
    int *rIdx;      // Indici delle righe originali (opzionale)
} HLLMatrix;

HLLMatrix *convert_csr_to_hll(CSRMatrix *csr);
void print_hll_matrix(HLLMatrix *hll);
void free_hll(HLLMatrix *hll);

#endif  // HLL_MATRIX_H