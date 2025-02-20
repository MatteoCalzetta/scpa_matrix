CC = gcc
CFLAGS = -Iinclude -Wall -Wextra
SRC = src/main.c src/csr_matrix.c
OUT = progetto.out

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)