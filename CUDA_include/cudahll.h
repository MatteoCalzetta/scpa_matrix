#ifndef CUDAHLL_H
#define CUDAHLL_H

#include "../include/hll_matrix.h"
#include "../include/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

double spmvHLL_CUDA(const HLLMatrix *hll_matrix, const double *d_x, double *d_y);
void calculate_nz_stats(const int *nz_per_row, int start_row, int end_row, int *max_nz, int *total_nz);

struct matrixPerformance parallel_hll_cuda_v1(HLLMatrix *hllMatrixHost, double *x_h);
struct matrixPerformance parallel_hll_cuda_v2(HLLMatrix *hllMatrixHost, double *x_h, double *y_h);
struct matrixPerformance parallel_hll_cuda_v3(HLLMatrix *hllMatrixHost, double *x_h, double *y_h);
struct matrixPerformance parallel_hll_column_cuda(const HLLMatrix *hll, const double *x_h, double *y_h);

#ifdef __cplusplus
}
#endif

#endif
