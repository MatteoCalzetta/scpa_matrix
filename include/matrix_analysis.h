#ifndef MATRIX_ANALYSIS_H
#define MATRIX_ANALYSIS_H

#include <stdbool.h>
#include "csr_matrix.h"

bool is_symmetric(CSRMatrix *csr);
bool is_triangular_upper(CSRMatrix *csr);
bool is_triangular_lower(CSRMatrix *csr);
bool is_diagonal(CSRMatrix *csr);
bool is_block_matrix(CSRMatrix *csr, int block_size);

#endif
