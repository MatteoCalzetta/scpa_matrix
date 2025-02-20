#include <stdio.h>
#include <stdlib.h>

// Funzione per leggere file MatrixMarket e convertire in CSR
void read_matrix_market(const char *filename, int *M, int *N, int *NZ, 
                        int **IRP, int **JA, double **AS) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    // Lettura intestazione e dimensioni della matrice
    fscanf(file, "%d %d %d", M, N, NZ);

    *IRP = (int *)malloc((*M + 1) * sizeof(int));
    *JA = (int *)malloc(*NZ * sizeof(int));
    *AS = (double *)malloc(*NZ * sizeof(double));

    int *row_count = (int *)calloc(*M, sizeof(int));

    // Leggere le righe e colonne della matrice
    for (int i = 0; i < *NZ; i++) {
        int row, col;
        double value;
        fscanf(file, "%d %d %lf", &row, &col, &value);
        row--; 
        col;