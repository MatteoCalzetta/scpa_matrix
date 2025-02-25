CC = gcc-14
CFLAGS = -Iinclude -I/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/aarch64-apple-darwin24/14/include -Wall -Wextra -fopenmp -O0
LDFLAGS = -fopenmp
SRC = src/main.c src/csr_matrix.c src/matrmult.c src/matrix_analysis.c src/openMP_prim.c src/json_results.c
OUT = progetto.out

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)

run: all
	./$(OUT)