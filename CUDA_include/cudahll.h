#ifndef CUDAHLL_H
#define CUDAHLL_H

#include "../include/hll_matrix.h"
#include "../include/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

double spmvHLL_CUDA(const HLLMatrix *hll_matrix, const double *d_x, double *d_y);
struct matrixPerformance parallel_hll_cuda_v1(HLLMatrix *hllMatrixHost, double *x_h);
void calculate_nz_stats(const int *nz_per_row, int start_row, int end_row, int *max_nz, int *total_nz);
struct matrixPerformance parallel_hll_cuda_shared(HLLMatrix *hllMatrixHost, double *x_h, double *y_h);
//void matvec_Hll_cuda_shared(const HLLMatrix *d_hll_matrix, const double *d_x, double *d_y, int M);

#ifdef __cplusplus
}
#endif

#endif
