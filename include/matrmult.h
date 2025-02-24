#ifndef MATRMULT_H
#define MATRMULT_H

#include "csr_matrix.h"

double csr_matrtimesvect(CSRMatrix *csr, int *x, double *y);
#endif
