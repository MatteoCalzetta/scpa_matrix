# Impostazione predefinita per il sistema
UNAME_S := $(shell uname -s)

# Configurazione per Ubuntu (GCC 13.3.0)
ifeq ($(UNAME_S), Linux)
  CC = gcc-13
  CFLAGS = -Iinclude -Wall -Wextra -fopenmp -O0
  LDFLAGS = -fopenmp
  # Puoi anche aggiungere eventuali altre configurazioni specifiche per Ubuntu
endif

# Configurazione per Apple (GCC 14)
ifeq ($(UNAME_S), Darwin)
	CC = gcc-14
	CFLAGS = -Iinclude -I/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/aarch64-apple-darwin24/14/include -Wall -Wextra -fopenmp -O0
	LDFLAGS = -fopenmp
# Puoi anche aggiungere eventuali altre configurazioni specifiche per Apple
endif

SRC = src/main.c src/csr_matrix.c src/matrmult.c src/matrix_analysis.c src/openMP_prim.c src/json_results.c src/hll_matrix.c src/mmio.c
OUT = progetto.out

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)

run: all
	./$(OUT)