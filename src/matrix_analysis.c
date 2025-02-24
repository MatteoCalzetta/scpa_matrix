#include "../include/matrix_analysis.h"
#include "../include/csr_matrix.h"
#include <stdio.h>
#include <stdbool.h>

// Controllo se la matrice è simmetrica in formato CSR
bool is_symmetric(CSRMatrix *csr) {
    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            int col = csr->JA[j]; // Colonna attuale
            double val = csr->AS[j]; // Valore attuale
            
            // Cerchiamo il valore speculare A[col, i]
            bool found = false;
            for (int k = csr->IRP[col]; k < csr->IRP[col + 1]; k++) {
                if (csr->JA[k] == i) {
                    if (csr->AS[k] != val) {
                        return false; // Se A[i, j] ≠ A[j, i], non è simmetrica
                    }
                    found = true;
                    break;
                }
            }
            if (!found) return false; // Se non troviamo il valore speculare, non è simmetrica
        }
    }
    return true;
}

// Controllo se la matrice è triangolare superiore in CSR
bool is_triangular_upper(CSRMatrix *csr) {
    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            if (csr->JA[j] < i) { 
                return false; // Se esiste un valore sotto la diagonale, non è triangolare superiore
            }
        }
    }
    return true;
}

// Controllo se la matrice è triangolare inferiore in CSR
bool is_triangular_lower(CSRMatrix *csr) {
    for (int i = 0; i < csr->M; i++) {
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            if (csr->JA[j] > i) { 
                return false; // Se esiste un valore sopra la diagonale, non è triangolare inferiore
            }
        }
    }
    return true;
}

// Controllo se la matrice è diagonale in CSR
bool is_diagonal(CSRMatrix *csr) {
    for (int i = 0; i < csr->M; i++) {
        if (csr->IRP[i + 1] - csr->IRP[i] != 1) {  
            return false; // Se c'è più di un elemento per riga, non è diagonale
        }
        int col = csr->JA[csr->IRP[i]];
        if (col != i) {  
            return false; // Se l'elemento non è sulla diagonale, non è diagonale
        }
    }
    return true;
}

// Controllo se la matrice è a blocchi in CSR
bool is_block_matrix(CSRMatrix *csr, int block_size) {
    if (csr->M % block_size != 0 || csr->N % block_size != 0) {
        return false; // La matrice non è divisibile in blocchi di questa dimensione
    }

    for (int bi = 0; bi < csr->M; bi += block_size) { // Per ogni blocco
        for (int i = bi; i < bi + block_size; i++) {
            for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
                int col = csr->JA[j];
                if (col < bi || col >= bi + block_size) {
                    return false; // Se un elemento non è nel blocco, non è a blocchi
                }
            }
        }
    }
    return true;
}