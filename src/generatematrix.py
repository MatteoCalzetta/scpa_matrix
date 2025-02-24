import random

rows = 1000
cols = 1000
nnz_per_row = 40  # Numero di elementi non nulli per riga
total_nnz = rows * nnz_per_row

filename = "test_matrix.mtx"

with open(filename, "w") as f:
    f.write("%%MatrixMarket matrix coordinate real general\n")
    f.write(f"% Test matrix for CSR conversion\n")
    f.write(f"{rows} {cols} {total_nnz}\n")

    for i in range(1, rows + 1):
        cols_selected = random.sample(range(1, cols + 1), nnz_per_row)
        for j in cols_selected:
            value = round(random.uniform(10, 100), 1)
            f.write(f"{i} {j} {value}\n")

print(f"Matrice 400x400 salvata in {filename}")