#ifndef CUDASRC_H
#define CUDASRC_H

#include "../include/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

double csr_matvec_cuda(CSRMatrix *csr, double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
