#ifndef HLL_CUDA_PRODUCT_H
#define HLL_CUDA_PRODUCT_H

#include "../include/hll_matrix.h"

// Funzione principale per la moltiplicazione matrice-vettore con HLL in CUDA
void hll_matrix_vector_cuda(const HLLMatrix *hll, const double *x, double *result);

#endif // HLL_CUDA_PRODUCT_H