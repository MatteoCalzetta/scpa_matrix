#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr_matrix.h"

// Implementazione delle funzioni...

void free_csr(CSRMatrix *csr) {
    free(csr->AS);
    free(csr->JA);
    free(csr->IRP);
    free(csr);
}