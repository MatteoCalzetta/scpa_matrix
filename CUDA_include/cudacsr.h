#ifndef CUDASRC_H
#define CUDASRC_H

#include "../include/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

double csr_matvec_cuda(CSRMatrix *csr, double *x, double *y);
double spmv_csr_threads(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_shmem(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_shmem_ridpar(CSRMatrix *csr, double *x, double *y);
double spmv_csr_warps_shmem_ridpar_launcher(CSRMatrix *csr, double *x, double *y);
double spmv_csr_gpu_texture(CSRMatrix *csr, const double *x, double *y);


#ifdef __cplusplus
}
#endif

#endif
