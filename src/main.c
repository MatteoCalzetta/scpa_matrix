#include <stdio.h>
#include "../include/csr_matrix.h"

int main() {
    const char *filename = "test_matrix.mtx";  // Nome del file da testare

    // Legge la matrice da file e converte in CSR
    CSRMatrix *csr = read_matrix_market(filename);

    if (csr) {
        printf("\n✅ Conversione in CSR completata con successo!\n\n");
        print_csr(csr);  // Stampa la matrice convertita in CSR
        free_csr(csr);   // Libera la memoria allocata
    } else {
        printf("\n❌ Errore nella lettura del file %s\n", filename);
    }

    return 0;
}