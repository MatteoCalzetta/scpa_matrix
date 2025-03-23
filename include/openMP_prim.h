#ifndef OPENMP_PRIM_H
#define OPENMP_PRIM_H

#include "csr_matrix.h"
#include <omp.h>

void csr_matvec_openmp(CSRMatrix *csr, double *x, double *y, int num_threads, int *row_partition);
void balance_load(CSRMatrix *csr, int num_threads, int *row_partition);
void hll_matvec_openmp(HLLMatrix *hll_matrix, double *x, double *y, int num_threads);

#endif