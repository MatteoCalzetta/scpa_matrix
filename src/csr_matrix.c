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

    // Controlliamo se la matrice è array (densa) o coordinate (sparsa)
    int is_sparse = mm_is_sparse(matcode);
    int is_array = mm_is_array(matcode);

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
    } else if (is_array) {
        if (mm_read_mtx_array_size(file, &M, &N) != 0) {
            printf("Errore nella lettura della dimensione della matrice array!\n");
            fclose(file);
            return NULL;
        }
        NZ = M * N; // Se è array, la matrice è densa
    } else {
        printf("Formato non riconosciuto!\n");
        fclose(file);
        return NULL;
    }

    // Allocazione per CSR
    int *IRP = calloc(M + 1, sizeof(int));
    int *JA = malloc(NZ * sizeof(int));
    double *AS = malloc(NZ * sizeof(double));

    if (!IRP || !JA || !AS) {
        printf("Errore di allocazione per CSR!\n");
        free(IRP); free(JA); free(AS);
        fclose(file);
        return NULL;
    }

    if (is_sparse) {
        int *I = malloc(NZ * sizeof(int));
        int *J = malloc(NZ * sizeof(int));
        double *values = malloc(NZ * sizeof(double));

        if (!I || !J || !values) {
            printf("Errore di allocazione della memoria!\n");
            free(I); free(J); free(values);
            free(IRP); free(JA); free(AS);
            fclose(file);
            return NULL;
        }

        for (int i = 0; i < NZ; i++) {
            fscanf(file, "%d %d %lf", &I[i], &J[i], &values[i]);
            I[i]--;  // Converti da 1-based a 0-based
            J[i]--;
        }

        // Conta gli elementi per riga
        for (int i = 0; i < NZ; i++) {
            IRP[I[i] + 1]++;
        }

        // Creazione di IRP
        for (int i = 1; i <= M; i++) {
            IRP[i] += IRP[i - 1];
        }

        // Costruzione degli array JA e AS
        int *row_fill = calloc(M, sizeof(int));
        for (int i = 0; i < NZ; i++) {
            int row = I[i];
            int pos = IRP[row] + row_fill[row];
            JA[pos] = J[i];
            AS[pos] = values[i];
            row_fill[row]++;
        }

        free(I);
        free(J);
        free(values);
        free(row_fill);

    } else if (is_array) {
        // Gestione del caso array: la matrice è densa
        int index = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double val;
                fscanf(file, "%lf", &val);
                if (val != 0) {  // Solo se diverso da zero
                    AS[index] = val;
                    JA[index] = j;
                    index++;
                }
            }
            IRP[i + 1] = index;
        }
        NZ = index;  // Aggiorniamo il numero di elementi non nulli
    }

    fclose(file);

    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!csr) {
        printf("Errore di allocazione per CSRMatrix!\n");
        free(IRP); free(JA); free(AS);
        return NULL;
    }

    csr->M = M;
    csr->N = N;
    csr->NZ = NZ;
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

