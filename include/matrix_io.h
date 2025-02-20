#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct {
    int M, N, NZ;   // Numero di righe, colonne e valori non zero
    double *AS;     // Valori non nulli
    int *JA;        // Indici delle colonne
    int *IRP;       // Puntatori di riga
} CSRMatrix;

CSRMatrix* read_matrix_market(const char *filename);
void print_csr(CSRMatrix *csr);
void free_csr(CSRMatrix *csr);

#endif