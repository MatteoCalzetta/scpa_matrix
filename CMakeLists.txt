cmake_minimum_required(VERSION 3.10)
project(progetto LANGUAGES C CXX CUDA)

# Imposta le architetture CUDA (puoi cambiare in base alla GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Opzioni di compilazione
set(CMAKE_CXX_FLAGS "-O3 -fopenmp -lm")
set(CMAKE_CUDA_FLAGS "-O3 -arch=sm_75 --compiler-options -fopenmp")

# Trova OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# Trova CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA include directory: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

# File sorgenti C
set(SRC_C
    src/main.c
    src/csr_matrix.c
    src/matrmult.c
    src/matrix_analysis.c
    src/openMP_prim.c
    src/json_results.c
    src/hll_matrix.c
    src/mmio.c
)

# File sorgenti CUDA
set(SRC_CUDA
    CUDA_src/cudacsr.cu
)

# Aggiunge i file CUDA come una libreria separata
cuda_add_library(cuda_kernels ${SRC_CUDA})

# Crea l'eseguibile
add_executable(progetto.out ${SRC_C})

# Collega librerie
target_link_libraries(progetto.out 
    cuda_kernels 
    ${CUDA_LIBRARIES} 
    OpenMP::OpenMP_CXX
)

# Includi directory per CUDA
target_include_directories(progetto.out PUBLIC ${CUDA_INCLUDE_DIRS})
