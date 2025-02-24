CC = gcc
CFLAGS = -Iinclude -Wall -Wextra
SRC = src/main.c src/csr_matrix.c src/matrmult.c src/matrix_analysis.c
OUT = progetto.out

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)

run: all
	./$(OUT)