cmake_minimum_required(VERSION 3.10)
project(progetto LANGUAGES C CXX CUDA)

# Imposta i compilatori per C e C++
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# Specifica l'architettura CUDA (per Quadro RTX 5000: 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Impostazioni di ottimizzazione e flag per OpenMP (C e C++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Iinclude -O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -lm")

# Imposta i flag per il linker
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

# Flag per CUDA, includendo l'opzione per OpenMP
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options=-fopenmp")

# Opzioni aggiuntive
add_compile_options(-ffast-math -funroll-loops)

# Trova e configura OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# Aggiunge i path ai CUDA Samples (se necessario)
set(CUDA_SAMPLES_PATH /home/mcalzetta/cuda-samples/Common)
set(CUDA_SAMPLES_PATH /home/alori/cuda-samples/Common)
include_directories(${CUDA_SAMPLES_PATH})

# File sorgenti (C, C++ e CUDA)
set(SRC_FILES
    src/main.c
    src/csr_matrix.c
    src/matrmult.c
    src/openMP_prim.c
    src/json_results.c
    src/hll_matrix.c
    src/mmio.c
    CUDA_src/cudacsr.cu
    CUDA_src/cudahll.cu
)

# Crea l'eseguibile
add_executable(progetto.out ${SRC_FILES})

# Linka OpenMP
target_link_libraries(progetto.out OpenMP::OpenMP_CXX)

# Abilita la compilazione separabile per CUDA
set_target_properties(progetto.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
