#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"


double matvec_Hll_serial(const HLLMatrix *hll_matrix, const double *x, double *y) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Avvia il timer

    for (int blockID = 0; blockID < hll_matrix->num_blocks; blockID++) {
        int start_row = blockID * HackSize;
        int end_row = (blockID + 1) * HackSize;
        if (end_row > hll_matrix->M) end_row = hll_matrix->M;  // Protezione extra

        int max_nz_per_row = hll_matrix->blocks[blockID].max_nz_per_row;
        int row_offset = 0;  // Inizializza l'offset per la riga corrente

        // Scorri le righe del blocco
        for (int i = start_row; i < end_row; i++) {
            y[i] = 0.0;

            // Scorri i non-zeri della riga
            for (int j = 0; j < max_nz_per_row; j++) {
                int idx = row_offset + j;
                if (hll_matrix->blocks[blockID].JA[idx] != -1) { // Salta il padding
                    y[i] += hll_matrix->blocks[blockID].AS[idx] * x[hll_matrix->blocks[blockID].JA[idx]];
                }
            }
            // Aggiorna l'offset per la prossima riga
            row_offset += max_nz_per_row;
        }
    }

    end = clock();  // Ferma il timer

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calcola il tempo in secondi

    return cpu_time_used;
}


// Funzione per il prodotto matrice-vettore BASE in CSR
double csr_matrtimesvect(CSRMatrix *csr, double *x, double *y) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Avvia il timer

    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            y[i] += csr->AS[j] * x[csr->JA[j]];
        }
    }

    end = clock();  // Ferma il timer

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calcola il tempo in secondi

    return cpu_time_used;
}



void matvec_Hll_serial_column_major(const HLLMatrix *hll_matrix, const double *x, double *y) {
    int M = hll_matrix->M;
    // Inizializza il vettore risultato
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
    }
    
    // Itera su ciascun blocco HLL
    for (int b = 0; b < hll_matrix->num_blocks; b++) {
        int start_row = b * HackSize;
        int rows_in_block = HackSize;
        // Per l'ultimo blocco, il numero di righe può essere inferiore a HackSize
        if (b == hll_matrix->num_blocks - 1) {
            int rem = hll_matrix->M % HackSize;
            if (rem != 0) {
                rows_in_block = rem;
            }
        }
        int max_nz_per_row = hll_matrix->blocks[b].max_nz_per_row;
        
        // Per ogni riga nel blocco
        for (int r = 0; r < rows_in_block; r++) {
            double sum = 0.0;
            // Per ogni "colonna" nel blocco (cioè per ogni posizione all'interno della riga, in formato column-major)
            for (int c = 0; c < max_nz_per_row; c++) {
                // In column-major l'elemento (r, c) si trova a:
                // index = c * rows_in_block + r
                int idx = c * rows_in_block + r;
                int col = hll_matrix->blocks[b].JA[idx];
                double val = hll_matrix->blocks[b].AS[idx];
                sum += val * x[col];
            }
            y[start_row + r] = sum;
        }
    }
}
