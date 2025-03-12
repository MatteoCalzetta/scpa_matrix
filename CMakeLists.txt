cmake_minimum_required(VERSION 3.10)
project(progetto LANGUAGES C CXX CUDA)

# Specifica l'architettura CUDA (per Quadro RTX 5000: 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Opzioni di ottimizzazione generali
add_compile_options(-O3 -march=native)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options -fopenmp")
#add_compile_options(-ffast-math) # Optimization Flag
#add_compile_options(-funroll-loops) # Optimization Flag

# Trova OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# File sorgenti C e CUDA
set(SRC_FILES
    src/main.c
    src/csr_matrix.c
    src/matrmult.c
    src/matrix_analysis.c
    src/openMP_prim.c
    src/json_results.c
    src/hll_matrix.c
    src/mmio.c
    CUDA_src/cudacsr.cu  # CUDA incluso direttamente
    CUDA_src/cudahll1d.cu
)

# Creazione dell'eseguibile con i file CUDA inclusi
add_executable(progetto.out ${SRC_FILES})

# Link a OpenMP
target_link_libraries(progetto.out OpenMP::OpenMP_CXX)

# Abilita CUDA per l'eseguibile
set_target_properties(progetto.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
