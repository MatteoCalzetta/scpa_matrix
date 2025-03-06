#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct {
    int M, N, NZ;
    double *AS;
    int *JA;
    int *IRP;
} CSRMatrix;

CSRMatrix* read_matrix_market(const char *filename);
void print_csr(CSRMatrix *csr);
void free_csr(CSRMatrix *csr);

#endif