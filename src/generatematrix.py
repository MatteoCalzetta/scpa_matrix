import random

# Parametri della matrice
rows = 20
cols = 20
nnz_per_row = 5  # Numero di elementi non nulli per riga
total_nnz = rows * nnz_per_row

# Scegli il tipo di matrice
matrix_type = input("Scegli il tipo di matrice (general/symmetric/pattern): ").strip().lower()

# Nome del file di output
filename = "test_matrix.mtx"

with open(filename, "w") as f:
    # Scrive l'intestazione Matrix Market
    if matrix_type == "symmetric":
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
    elif matrix_type == "pattern":
        f.write("%%MatrixMarket matrix coordinate pattern general\n")
    else:
        f.write("%%MatrixMarket matrix coordinate real general\n")

    f.write("% Test matrix for CSR conversion\n")
    
    # Lista per salvare gli elementi
    elements = set()

    for i in range(1, rows + 1):
        cols_selected = random.sample(range(1, cols + 1), nnz_per_row)
        for j in cols_selected:
            if matrix_type == "pattern":
                value = 1.0
            else:
                value = round(random.uniform(10, 100), 1)
            
            if matrix_type == "symmetric" and i > j:
                continue  # Salva solo il triangolo superiore
            
            elements.add((i, j, value))

    # Se simmetrica, aggiunge gli elementi speculari
    if matrix_type == "symmetric":
        extra_elements = set((j, i, v) for i, j, v in elements if i != j)
        elements.update(extra_elements)

    # Scrive dimensioni e numero di nonzeri
    f.write(f"{rows} {cols} {len(elements)}\n")

    # Scrive gli elementi della matrice
    for i, j, value in sorted(elements):
        if matrix_type == "pattern":
            f.write(f"{i} {j}\n")  # Nessun valore scritto
        else:
            f.write(f"{i} {j} {value}\n")

print(f"Matrice {matrix_type} salvata in {filename}")