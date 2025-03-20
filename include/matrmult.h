#ifndef MATRMULT_H
#define MATRMULT_H

#include "csr_matrix.h"
#include "hll_matrix.h"


double csr_matrtimesvect(CSRMatrix *csr, double *x, double *y);
double matvec_Hll_serial(const HLLMatrix *hll_matrix, const double *x, double *y);
void matvec_Hll_serial_column_major(const HLLMatrix *hll_matrix, const double *x, double *y);
#endif
