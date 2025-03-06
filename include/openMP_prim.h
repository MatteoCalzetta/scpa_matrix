#ifndef OPENMP_PRIM_H
#define OPENMP_PRIM_H

#include "csr_matrix.h"
#include <omp.h>

double csr_matvec_openmp(CSRMatrix *csr, double *x, double *y, int num_threads, int *row_partition);
void balance_load(CSRMatrix *csr, int num_threads, int *row_partition);

#endif