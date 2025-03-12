#ifndef CUDAHLL1D_H
#define CUDAHLL1D_H

#include "../include/hll_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void spmvHLL_CUDA(HLLMatrix *hll, const double *h_x, double *h_y);

#ifdef __cplusplus
}
#endif

#endif