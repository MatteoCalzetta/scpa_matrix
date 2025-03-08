# Rilevamento del sistema operativo
UNAME_S := $(shell uname -s)

# Configurazione per Ubuntu (GCC 13.3.0)
ifeq ($(UNAME_S), Linux)
  CC = gcc
  CFLAGS = -Iinclude -Wall -Wextra -fopenmp -O3
  LDFLAGS = -fopenmp
  NVCC = nvcc
  NVCCFLAGS = -Iinclude -O3 -arch=sm_75
  
endif

# Configurazione per macOS (GCC 14)
ifeq ($(UNAME_S), Darwin)
  CC = gcc-14
  CFLAGS = -Iinclude -I/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/aarch64-apple-darwin24/14/include -Wall -Wextra -fopenmp -O3
  LDFLAGS = -fopenmp
  NVCC = nvcc
  NVCCFLAGS = -Iinclude -O3 -arch=sm_75
endif

# File sorgenti
SRC_C = src/main.c src/csr_matrix.c src/matrmult.c src/matrix_analysis.c src/openMP_prim.c src/json_results.c src/hll_matrix.c src/mmio.c
SRC_CUDA = CUDA_src/cudacsr.cu
OUT = progetto.out

# Regole di compilazione
all: $(OUT)

$(OUT): $(SRC_C) $(SRC_CUDA)
	$(CC) $(CFLAGS) -c $(SRC_C)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_CUDA)
	$(CC) $(CFLAGS) -o $(OUT) *.o $(LDFLAGS) -lcudart

clean:
	rm -f $(OUT) *.o

run: all
	./$(OUT)
