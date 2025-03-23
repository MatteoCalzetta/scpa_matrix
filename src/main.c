#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "../include/json_results.h"
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"
#include "../include/openMP_prim.h"
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudacsr.h"
#include "../include/matrix.h"
#include "../CUDA_include/cudahll.h"

#define MATRIX_DIR "../build/test_matrix/"  

// Funzione per controllare la conversione da CSR a HLL (colonnare)
#include <math.h>
#include <stdio.h>
#include "../include/csr_matrix.h"
#include "../include/hll_matrix.h"

#define HACK_SIZE 32

void generate_random_vector(double *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (rand() % 5) + 1; 
    }
}

double compute_norm(double *store, double *y, int M) {
    double norm = 0.0;
    for (int i = 0; i < M; i++) {
        double diff = store[i] - y[i];
        norm += diff * diff;
    }
    return sqrt(norm);
}

int main() {
    int thread_counts[] = {2, 4, 8, 16, 32, 40};
    omp_set_dynamic(0);
    srand(time(NULL));

    MatrixResult results[num_matrices];

    for (int i = 0; i < num_matrices; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s%s", MATRIX_DIR, matrix_filenames[i]);

        // Legge la matrice e converte in CSR
        CSRMatrix *csr = read_matrix_market(filename);
        if (!csr) {
            printf("Errore: impossibile leggere il file %s\n", filename);
            continue;
        }
        
        // Conversione diretta da Matrix Market a HLL
        HLLMatrix *hll = convert_csr_to_hll(csr);
        //print_hll_matrix_csr(hll);

        double *x = (double *)malloc(csr->N * sizeof(double));

        double *y2 = (double *)calloc(csr->M, sizeof(double));

        double *serial_csr = (double *)calloc(csr->M, sizeof(double));
        double *serial_hll = (double *)calloc(csr->M, sizeof(double));
        double *omp_csr = (double *)calloc(csr->M, sizeof(double));
        double *cuda_hll = (double *)calloc(csr->M, sizeof(double));

        double *omp_hll = (double *)calloc(csr->M, sizeof(double));

        double *serial_static_hll = (double *)calloc(csr->M, sizeof(double));
        double *hll_cuda_k2 = (double *)calloc(csr->M, sizeof(double));


        if (!x || !serial_csr || !serial_hll || !omp_csr || !cuda_hll) {
            printf("Errore: allocazione fallita per i vettori x o y (file: %s)\n", filename);
            free_csr(csr);
            continue;
        }
        generate_random_vector(x, csr->N);

        snprintf(results[i].matrix_name, sizeof(results[i].matrix_name), "%s", matrix_filenames[i]);
        
        
        // **Esecuzione Seriale CSR**
        results[i].serial.time = csr_matrtimesvect(csr, x, serial_csr);
        //printf("[Seriale CSR] Matrice: %s | Tempo: %.10f s\n", matrix_filenames[i], results[i].serial.time);

        // **Esecuzione Seriale HLL**
        double result = matvec_Hll_serial(hll, x, serial_hll);
        printf("[Seriale HLL] Matrice: %s | Tempo: %.10f s\n", matrix_filenames[i], result);
        printf("Norma tra seriale HLL e CSR = %.4f\n", compute_norm(serial_csr, serial_hll, csr->M));

        // **Esecuzione cuda HLL** 
        /*struct matrixPerformance result_hll_cuda = parallel_hll_cuda_v1(hll, x);
        result_hll_cuda.gigaFlops = (2.0 * csr->NZ) / (result_hll_cuda.seconds * 1e9);
        printf("CUDA HLL Performance: %.4f GigaFLOPS\n", result_hll_cuda.gigaFlops);
        printf("Norma tra seriale HLL e CSR = %.4f\n", compute_norm(serial_csr, serial_hll, csr->M));*/


        // **Esecuzione CUDA**
        /*float total_cuda_time = 0.0;
        int repetitions = 100;
        for (int r = 0; r < repetitions; r++) {
            total_cuda_time += spmvHLL_CUDA(hll, x, y);
        }
        double cuda_result = total_cuda_time / repetitions;
        printf("[CUDA HLL] Matrice: %s | Tempo medio: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], cuda_result, (2.0 * csr->NZ) / (cuda_result * 1e9));
        double cuda_norm = compute_norm(store, y, csr->M);
        printf("Norma L2 tra seriale e CUDA HLL: %.4f\n\n", cuda_norm);*/



        /*
        // CUDA Thread per riga kernel0
        results[i].cuda.time = spmv_csr_threads(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / results[i].cuda.time;
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        double norm_value = compute_norm(store, y2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        */

        
        // CUDA Warp per riga con __shfl_sync kernel1
        results[i].cuda.time = spmv_csr_warps(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / (results[i].cuda.time * 1e9);
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | GFLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        double norm_value = compute_norm(serial_csr, y2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        
        
        
        /*
        // CUDA Warp per riga con shared_memory kernel2
        results[i].cuda.time = spmv_csr_warps_shmem(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / results[i].cuda.time;
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        double norm_value = compute_norm(serial_csr, y2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        */

        /*
        // CUDA Warp per riga con shared_memory e rid parall kernel3
        results[i].cuda.time = spmv_csr_warps_shmem_ridpar(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / results[i].cuda.time;
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        double norm_value = compute_norm(serial_csr, y2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        */

        
        /*
        // PROVA K4
        results[i].cuda.time = spmv_csr_warps_shmem_ridpar_launcher(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / results[i].cuda.time;
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        double norm_value = compute_norm(serial_csr, y2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        */


        /*
        // PROVA K5
        results[i].cuda.time = spmv_csr_gpu_texture(csr, x, y2);
        results[i].cuda.flops = (2.0 * csr->NZ) / results[i].cuda.time;
        printf("[CUDA] Matrice: %s | Tempo: %.10f s | FLOPS: %.10f\n", matrix_filenames[i], results[i].cuda.time, results[i].cuda.flops);
        double norm_value = compute_norm(serial_csr, y2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        */

        struct matrixPerformance perf2 = parallel_hll_cuda_v2(hll, x, hll_cuda_k2);
        printf("[CUDA HLL KERNEL 2] Matrice: %s | GFLOPS: %.10f\n", matrix_filenames[i], (2 * csr->NZ)/(perf2.seconds * 1e9));
        norm_value = compute_norm(serial_csr, hll_cuda_k2, csr->M);
        printf("Norma L2 tra seriale e CUDA: %f\n\n", norm_value);
        

        /*for (int i = 0; i < hll->num_blocks; i++) {
            // Per ogni blocco, il numero di righe è HACK_SIZE, salvo per l'ultimo blocco
            int rows_in_block = HACK_SIZE;
            if (i == hll->num_blocks - 1) {
                int r = hll->M % HACK_SIZE;
                if (r != 0) {
                    rows_in_block = r;
                }
            }
            convert_block_to_column_major(&hll->blocks[i], rows_in_block);
        }

        matvec_Hll_serial_column_major(hll, x, serial_static_hll);

        struct matrixPerformance perf = parallel_hll_cuda_shared(hll, x, cuda_hll);
        printf("[CUDA HLL TRASPOSTO] Matrice: %s | FLOPS: %.10f\n", matrix_filenames[i], (2 * csr->NZ)/(perf.seconds * 1e9));
        printf("Norma tra seriale HLL Trasposto e CSR = %.4f\n", compute_norm(serial_csr, cuda_hll, csr->M));*/
        

        // **Esecuzione OpenMP**
        results[i].num_openmp = 0;
        for (int j = 0; j < (sizeof(thread_counts) / sizeof(int)); j++) {


            /*
            if (hll->M < thread_counts[j]) continue;
            

            results[i].openmp_results[results[i].num_openmp].threads = thread_counts[j];
            omp_set_num_threads(thread_counts[j]);
            memset(omp_hll, 0, hll->M * sizeof(double));
            double start_time = omp_get_wtime();
            hll_matvec_openmp(hll, x, omp_hll, thread_counts[j]);
            double end_time = omp_get_wtime();
            double norm_value = compute_norm(serial_csr, omp_hll, hll->M);
            printf("Norma L2 tra seriale e hll: %f\n", norm_value);
            results[i].openmp_results[results[i].num_openmp].time = end_time-start_time;
            results[i].openmp_results[results[i].num_openmp].flops = (2.0 * csr->NZ) / (results[i].openmp_results[results[i].num_openmp].time * 1e9);
            results[i].num_openmp++;
            
            printf("[OpenMP] Matrice: %s | Threads: %d | FLOPS: %.10f | Tempo: %.10f s | NZ: %d\n", 
                   matrix_filenames[i], thread_counts[j], results[i].openmp_results[results[i].num_openmp-1].flops, results[i].openmp_results[results[i].num_openmp-1].time, csr->NZ);
            */




            if (csr->M < thread_counts[j]) continue;
            
            int *row_partition = (int *)malloc((thread_counts[j]+1) * sizeof(int));
            if (!row_partition) {
                printf("Errore: allocazione fallita per row_partition (file: %s, threads: %d)\n", filename, thread_counts[j]);
                free(x);
                free_csr(csr);
                continue;
            }

            balance_load(csr, thread_counts[j], row_partition);

            

            results[i].openmp_results[results[i].num_openmp].threads = thread_counts[j];
            //printf("numero di thread passato e' %d\n", thread_counts[j]);
            omp_set_num_threads(thread_counts[j]);
            memset(omp_csr, 0, csr->M * sizeof(double));
            double start_time = omp_get_wtime();
            csr_matvec_openmp(csr, x, omp_csr, thread_counts[j], row_partition);
            double norm_value = compute_norm(serial_csr, omp_csr, csr->M);
            //printf("Norma L2 tra seriale e openmp: %f\n", norm_value);
            double end_time = omp_get_wtime();
            results[i].openmp_results[results[i].num_openmp].time = end_time-start_time;
            results[i].openmp_results[results[i].num_openmp].flops = (2.0 * csr->NZ) / (results[i].openmp_results[results[i].num_openmp].time * 1e9);
            results[i].num_openmp++;
            
            //printf("[OpenMP] Matrice: %s | Threads: %d | FLOPS: %.10f | Tempo: %.10f s | NZ: %d\n", 
                   //matrix_filenames[i], thread_counts[j], results[i].openmp_results[results[i].num_openmp-1].flops, results[i].openmp_results[results[i].num_openmp-1].time, csr->NZ);
            
            free(row_partition);
        }

        free(x);
        free(serial_csr);
        free(serial_hll);
        free(omp_csr);
        free(cuda_hll);
        free(hll_cuda_k2);
        free_csr(csr);
        free_hll_matrix(hll);
    }

    // Scriviamo i risultati sul file JSON
    write_results_to_json("results.json", results, num_matrices);
    
    return 0;
}