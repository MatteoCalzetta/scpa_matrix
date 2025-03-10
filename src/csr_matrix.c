#include <stdio.h>
#include <stdlib.h>
#include "../include/mmio.h"
#include "../include/csr_matrix.h"

CSRMatrix *read_matrix_market(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Errore: impossibile aprire il file %s\n", filename);
        return NULL;
    }

    MM_typecode matcode;
    if (mm_read_banner(file, &matcode) != 0) {
        printf("Errore nella lettura del banner Matrix Market!\n");
        fclose(file);
        return NULL;
    }

    int is_sparse = mm_is_sparse(matcode);
    int is_symmetric = mm_is_symmetric(matcode);
    int is_pattern = mm_is_pattern(matcode);

    if (!mm_is_matrix(matcode) || !mm_is_real(matcode)) {
        printf("Formato non supportato! Deve essere una matrice reale.\n");
        fclose(file);
        return NULL;
    }

    int M, N, NZ;
    if (is_sparse) {
        if (mm_read_mtx_crd_size(file, &M, &N, &NZ) != 0) {
            printf("Errore nella lettura della dimensione della matrice!\n");
            fclose(file);
            return NULL;
        }
    } else {
        printf("Formato non supportato! La matrice deve essere in formato coordinate.\n");
        fclose(file);
        return NULL;
    }

    int maxNZ = is_symmetric ? NZ * 2 : NZ;
    int *IRP = calloc(M + 1, sizeof(int));
    int *JA = malloc(maxNZ * sizeof(int));
    double *AS = malloc(maxNZ * sizeof(double));

    if (!IRP || !JA || !AS) {
        printf("Errore di allocazione per CSR!\n");
        free(IRP); free(JA); free(AS);
        fclose(file);
        return NULL;
    }

    int *I = malloc(maxNZ * sizeof(int));
    int *J = malloc(maxNZ * sizeof(int));
    double *values = is_pattern ? NULL : malloc(maxNZ * sizeof(double));

    if (!I || !J || (!is_pattern && !values)) {
        printf("Errore di allocazione della memoria!\n");
        free(I); free(J); free(values);
        free(IRP); free(JA); free(AS);
        fclose(file);
        return NULL;
    }

    int count = 0;
    for (int i = 0; i < NZ; i++) {
        int row, col;
        double val = 1.0;  // Default per pattern
        if (is_pattern) {
            fscanf(file, "%d %d", &row, &col);
        } else {
            fscanf(file, "%d %d %lf", &row, &col, &val);
        }
        row--; col--;

        I[count] = row;
        J[count] = col;
        if (!is_pattern) values[count] = val;
        count++;

        if (is_symmetric && row != col) {
            I[count] = col;
            J[count] = row;
            if (!is_pattern) values[count] = val;
            count++;
        }
    }
    fclose(file);

    for (int i = 0; i < count; i++) {
        IRP[I[i] + 1]++;
    }

    for (int i = 1; i <= M; i++) {
        IRP[i] += IRP[i - 1];
    }

    int *row_fill = calloc(M, sizeof(int));
    for (int i = 0; i < count; i++) {
        int row = I[i];
        int pos = IRP[row] + row_fill[row];
        JA[pos] = J[i];
        AS[pos] = is_pattern ? 1.0 : values[i];
        row_fill[row]++;
    }

    free(I);
    free(J);
    free(values);
    free(row_fill);

    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!csr) {
        printf("Errore di allocazione per CSRMatrix!\n");
        free(IRP); free(JA); free(AS);
        return NULL;
    }

    csr->M = M;
    csr->N = N;
    csr->NZ = count;
    csr->AS = AS;
    csr->JA = JA;
    csr->IRP = IRP;

    return csr;
}


void print_csr(CSRMatrix *csr) {
    if (!csr) {
        printf("La matrice CSR è vuota!\n");
        return;
    }

    printf("Matrice in formato CSR:\n");
    printf("Dimensioni: %d x %d\n", csr->M, csr->N);
    printf("Numero di elementi non nulli: %d\n", csr->NZ);

    printf("\nAS (valori non nulli):\n");
    for (int i = 0; i < csr->NZ; i++) {
        printf("%.10f ", csr->AS[i]);
    }
    printf("\n");

    printf("\nJA (indici di colonna):\n");
    for (int i = 0; i < csr->NZ; i++) {
        printf("%d ", csr->JA[i]);
    }
    printf("\n");

    printf("\nIRP (puntatori di riga):\n");
    for (int i = 0; i <= csr->M; i++) {
        printf("%d ", csr->IRP[i]);
    }
    printf("\n");
}



void free_csr(CSRMatrix *csr) {
    if (!csr) return;
    free(csr->AS);
    free(csr->JA);
    free(csr->IRP);
    free(csr);
}

