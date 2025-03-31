#ifndef CUDASRC_H
#define CUDASRC_H

#include "../include/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

double spmv_csr_threads(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_shmem(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_shmem_ridpar(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_cachel2(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_texture(CSRMatrix *csr, const double *x, double *y);


#ifdef __cplusplus
}
#endif

#endif
