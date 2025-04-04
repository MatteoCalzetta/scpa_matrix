#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// Header con MatrixResult e la write_results_to_json
#include "../include/json_results.h"

// Altri header del progetto
#include "../include/csr_matrix.h"
#include "../include/matrmult.h"
#include "../include/openMP_prim.h"
#include "../include/hll_matrix.h"
#include "../CUDA_include/cudacsr.h"
#include "../include/matrix.h"
#include "../CUDA_include/cudahll.h"

#define MATRIX_DIR "../build/test_matrix/"
#define HACK_SIZE 32

// Array di nomi di matrici e contatore (o esterni, vedi tu)
extern const char *matrix_filenames[]; 
extern const int num_matrices;

void spmv_serial_hll_column(const HLLMatrix *hll, const double *x, double *y) {
    for (int b = 0; b < hll->num_blocks; b++) {
        const HLLBlock *block = &hll->blocks[b];
        int rows = block->rows_in_block;
        int max_nz = block->max_nz_per_row;

        for (int r = 0; r < rows; r++) {
            double sum = 0.0;
            for (int c = 0; c < max_nz; c++) {
                int idx = c * rows + r;  // ← column-major access
                int col = block->JA[idx];
                double val = block->AS[idx];
                if (col >= 0) {
                    sum += val * x[col];
                }
            }
            int global_row = b * HACK_SIZE + r;
            y[global_row] = sum;
        }
    }
}

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

    // Array di risultati (uno per matrice)
    MatrixResult results[num_matrices];

    for (int i = 0; i < num_matrices; i++) {
        memset(&results[i], 0, sizeof(MatrixResult));

        // Prepara il nome del file e leggi la matrice
        char filename[256];
        snprintf(filename, sizeof(filename), "%s%s", MATRIX_DIR, matrix_filenames[i]);

        CSRMatrix *csr = read_matrix_market(filename);
        if (!csr) {
            printf("Errore: impossibile leggere il file %s\n", filename);
            continue;
        }

        results[i].nz = csr->NZ;

        // Converte in HLL
        HLLMatrix *hll = convert_csr_to_hll(csr);
        HLLMatrix *hll_column = convert_csr_to_hll_column_major(csr);

        // Alloca vettori
        double *x          = (double *)malloc(csr->N * sizeof(double));
        double *serial_csr = (double *)calloc(csr->M, sizeof(double));
        double *serial_hll = (double *)calloc(csr->M, sizeof(double));
        double *omp_csr    = (double *)calloc(csr->M, sizeof(double));
        double *omp_hll    = (double *)calloc(csr->M, sizeof(double));
        double *cuda_hll   = (double *)calloc(csr->M, sizeof(double));
        double *hll_cuda_k2= (double *)calloc(csr->M, sizeof(double));
        double *y2         = (double *)calloc(csr->M, sizeof(double)); // per i kernel CSR

        if (!x || !serial_csr || !serial_hll || !omp_csr || !omp_hll ||
            !cuda_hll || !hll_cuda_k2 || !y2) {
            printf("Errore: allocazione fallita per i vettori (file: %s)\n", filename);
            free_csr(csr);
            continue;
        }
        generate_random_vector(x, csr->N);

        for (int i = 0; i < csr->N; i++) x[i] = 1.0;

        // Salviamo il nome
        snprintf(results[i].matrix_name, sizeof(results[i].matrix_name),
                 "%s", matrix_filenames[i]);
        results[i].nz = csr->NZ;

        /*
         * ===========================================================
         *             1) CHIAMATE SERIALI (CSR e HLL)
         * ===========================================================
         */
        double time_csr_serial = csr_matrtimesvect(csr, x, serial_csr);
        double flops_csr_serial = (2.0 * csr->NZ) / (time_csr_serial * 1e9);
        results[i].serial.time  = time_csr_serial;
        results[i].serial.flops = flops_csr_serial;

        //printf("[SERIAL CSR] %s | Tempo: %.5f s | GFLOPS: %.5f\n", matrix_filenames[i], time_csr_serial, flops_csr_serial);

        double *y = (double *)calloc(csr->M, sizeof(double));
        spmv_serial_hll_column(hll_column, x, y);
        double norm_hll_col_vs_csr = compute_norm(serial_csr, y, csr->M);
        //printf("Norma L2 (CSR vs HLL_column) = %.4f\n\n", norm_hll_col_vs_csr);



        double time_hll_serial = matvec_Hll_serial(hll, x, serial_hll);
        double flops_hll_serial = (2.0 * csr->NZ)/(time_hll_serial * 1e9);
        //printf("[SERIAL HLL] %s | Tempo: %.5f s | GFLOPS: %.5f\n", matrix_filenames[i], time_hll_serial, flops_hll_serial);
        results[i].serial_hll.time = time_hll_serial;
        results[i].serial_hll.flops = flops_hll_serial;

        double norm_hll_vs_csr = compute_norm(serial_csr, serial_hll, csr->M);
        //printf("Norma L2 (CSR vs HLL) = %.4f\n\n", norm_hll_vs_csr);

        /*
         * ===========================================================
         *             2) CHIAMATE CUDA CSR
         * ===========================================================
         *  Salviamo OGNI KERNEL in results[i].cuda_csr[] 
         */
        // Reinizializza contatore
        results[i].num_cuda_csr = 0;

        // Kernel 0
        memset(y2, 0, csr->M * sizeof(double));
        double k0_time = spmv_csr_threads(csr, x, y2);
        double k0_flops = (2.0 * csr->NZ)/(k0_time * 1e9);
        // Aggiungo nel cuda_csr
        int idx_csr = results[i].num_cuda_csr++;
        strcpy(results[i].cuda_csr[idx_csr].kernel_name, "kernel0_threads");
        results[i].cuda_csr[idx_csr].time   = k0_time;
        results[i].cuda_csr[idx_csr].gflops = k0_flops;

        // Kernel 1
        memset(y2, 0, csr->M * sizeof(double));
        double k1_time = spmv_csr_warps(csr, x, y2);
        double k1_flops = (2.0 * csr->NZ)/(k1_time * 1e9);
        idx_csr = results[i].num_cuda_csr++;
        strcpy(results[i].cuda_csr[idx_csr].kernel_name, "kernel1_warps");
        results[i].cuda_csr[idx_csr].time   = k1_time;
        results[i].cuda_csr[idx_csr].gflops = k1_flops;

        // Kernel 2
        memset(y2, 0, csr->M * sizeof(double));
        double k2_time = spmv_csr_warps_shmem(csr, x, y2);
        double k2_flops = (2.0 * csr->NZ)/(k2_time * 1e9);
        idx_csr = results[i].num_cuda_csr++;
        strcpy(results[i].cuda_csr[idx_csr].kernel_name, "kernel2_warps_shmem");
        results[i].cuda_csr[idx_csr].time   = k2_time;
        results[i].cuda_csr[idx_csr].gflops = k2_flops;

        // Kernel 3
        memset(y2, 0, csr->M * sizeof(double));
        double k3_time = spmv_csr_warps_shmem_ridpar(csr, x, y2);
        double k3_flops = (2.0 * csr->NZ)/(k3_time * 1e9);
        idx_csr = results[i].num_cuda_csr++;
        strcpy(results[i].cuda_csr[idx_csr].kernel_name, "kernel3_warps_shmem_ridpar");
        results[i].cuda_csr[idx_csr].time   = k3_time;
        results[i].cuda_csr[idx_csr].gflops = k3_flops;

        // Kernel 4
        memset(y2, 0, csr->M * sizeof(double));
        double k4_time = spmv_csr_warps_cachel2(csr, x, y2);
        double k4_flops = (2.0 * csr->NZ)/(k4_time * 1e9);
        idx_csr = results[i].num_cuda_csr++;
        strcpy(results[i].cuda_csr[idx_csr].kernel_name, "kernel4_warps_cachel2");
        results[i].cuda_csr[idx_csr].time   = k4_time;
        results[i].cuda_csr[idx_csr].gflops = k4_flops;

        // Kernel 5
        memset(y2, 0, csr->M * sizeof(double));
        double k5_time = spmv_csr_warps_texture(csr, x, y2);
        double k5_flops = (2.0 * csr->NZ)/(k5_time * 1e9);
        idx_csr = results[i].num_cuda_csr++;
        strcpy(results[i].cuda_csr[idx_csr].kernel_name, "kernel5_texture");
        results[i].cuda_csr[idx_csr].time   = k5_time;
        results[i].cuda_csr[idx_csr].gflops = k5_flops;

        /*
         * ===========================================================
         *             3) CHIAMATE CUDA HLL
         * ===========================================================
         *  Salviamo OGNI kernel in results[i].cuda_hll[] 
         */
        results[i].num_cuda_hll = 0;

        /*
        // v1: kernel row major, un thread per riga.
        struct matrixPerformance r_v1 = parallel_hll_cuda_v1(hll, x);
        double v1_gflops = (2.0 * csr->NZ)/(r_v1.seconds * 1e9);
        int idx_hll = results[i].num_cuda_hll++;
        strcpy(results[i].cuda_hll[idx_hll].kernel_name, "hll_v1");
        results[i].cuda_hll[idx_hll].time   = r_v1.seconds;
        results[i].cuda_hll[idx_hll].gflops = v1_gflops;
        
        
        // v2 kernel row major, un warp per riga.
        struct matrixPerformance r_v2 = parallel_hll_cuda_v2(hll, x, hll_cuda_k2);
        double v2_gflops = (2.0 * csr->NZ)/(r_v2.seconds * 1e9);
        idx_hll = results[i].num_cuda_hll++;
        strcpy(results[i].cuda_hll[idx_hll].kernel_name, "hll_v2");
        results[i].cuda_hll[idx_hll].time   = r_v2.seconds;
        results[i].cuda_hll[idx_hll].gflops = v2_gflops;
        */
        
        // v3 kernel column-major, un thread per riga (accessi coalescenti op)
        memset(cuda_hll, 0, csr->M * sizeof(double));
        struct matrixPerformance r_hll_col = parallel_hll_column_cuda(hll_column, x, cuda_hll);
        double gflops_hll_col = (2.0 * csr->NZ) / (r_hll_col.seconds * 1e9);
        //printf("[CUDA HLL Column] %s | GFLOPS: %.5f\n", matrix_filenames[i], gflops_hll_col);

        //double norm_cuda_vs_serial = compute_norm(serial_csr, cuda_hll, csr->M);
        //printf("Norma L2 (CSR seriale vs CUDA HLL column) = %.4f\n\n", norm_cuda_vs_serial);

        int idx_hll_col = results[i].num_cuda_hll++;
        strcpy(results[i].cuda_hll[idx_hll_col].kernel_name, "hll_column_basic");
        results[i].cuda_hll[idx_hll_col].time   = r_hll_col.seconds;
        results[i].cuda_hll[idx_hll_col].gflops = gflops_hll_col;
        
        
        /*
        struct matrixPerformance r_v3 = parallel_hll_cuda_v3(hll, x, hll_cuda_k2);
        double v3_gflops = (2.0 * csr->NZ)/(r_v3.seconds * 1e9);
        idx_hll = results[i].num_cuda_hll++;
        strcpy(results[i].cuda_hll[idx_hll].kernel_name, "hll_v3");
        results[i].cuda_hll[idx_hll].time   = r_v3.seconds;
        results[i].cuda_hll[idx_hll].gflops = v3_gflops;
        //double norm_cuda_vs_serial = compute_norm(serial_csr, hll_cuda_k2, csr->M);
        //printf("Norma L2 (CSR seriale vs CUDA HLL column) = %.4f\n\n", norm_cuda_vs_serial);
        //printf("[CUDA HLL Column] %s | GFLOPS: %.5f\n", matrix_filenames[i], v3_gflops);
        */
        
        /*
         * ===========================================================
         *             4) CHIAMATE OPENMP
         * ===========================================================
         */
        results[i].num_openmp_csr = 0;
        results[i].num_openmp_hll = 0;

        for (int j = 0; j < (sizeof(thread_counts) / sizeof(int)); j++) {
            int n_threads = thread_counts[j];

            if (hll->M < n_threads) continue; // Evita HLL con troppi threads

            memset(omp_hll, 0, hll->M * sizeof(double));
            omp_set_num_threads(n_threads);
            double start_time = omp_get_wtime();
            hll_matvec_openmp(hll, x, omp_hll, n_threads);
            double end_time   = omp_get_wtime();

            double omp_time_hll  = end_time - start_time;
            double omp_flops_hll = (2.0 * csr->NZ)/(omp_time_hll * 1e9);
            int idx_omp = results[i].num_openmp_hll++;
            results[i].openmp_hll[idx_omp].threads = n_threads;
            results[i].openmp_hll[idx_omp].time    = omp_time_hll;
            results[i].openmp_hll[idx_omp].flops   = omp_flops_hll;
            results[i].openmp_hll[idx_omp].speedup = time_hll_serial/omp_time_hll;
            results[i].openmp_hll[idx_omp].efficienza = (time_hll_serial/omp_time_hll)/n_threads;        

            // CSR OpenMP
            if (csr->M < n_threads) continue;
            int actual_threads;
            int *row_partition = balance_load(csr, n_threads, &actual_threads);


            memset(omp_csr, 0, csr->M * sizeof(double));
            double t_start = omp_get_wtime();
            csr_matvec_openmp(csr, x, omp_csr, actual_threads, row_partition);
            double t_end   = omp_get_wtime();
            free(row_partition);

            double norm_cuda_vs_serial = compute_norm(serial_csr, omp_csr, csr->M);
            //printf("Norma L2 (CSR seriale vs CUDA HLL column) = %.4f\n\n", norm_cuda_vs_serial);

            double omp_time_csr  = t_end - t_start;
            double omp_flops_csr = (2.0 * csr->NZ)/(omp_time_csr * 1e9);
            idx_omp = results[i].num_openmp_csr++;
            results[i].openmp_csr[idx_omp].threads = n_threads;
            results[i].openmp_csr[idx_omp].time    = omp_time_csr;
            results[i].openmp_csr[idx_omp].flops   = omp_flops_csr;
            results[i].openmp_csr[idx_omp].speedup = time_csr_serial/omp_time_csr;
            results[i].openmp_csr[idx_omp].efficienza = (time_csr_serial/omp_time_csr)/n_threads;
         
        }

        // Deallocazioni finali
        free(x);
        free(serial_csr);
        free(serial_hll);
        free(omp_csr);
        free(omp_hll);
        free(cuda_hll);
        free(hll_cuda_k2);
        free(y2);
        free(y);
        free_csr(csr);
        free_hll_matrix(hll);
        free_hll_matrix_col(hll_column);
    }

    // Scriviamo tutto
    write_results_to_json("results.json", results, num_matrices);

    return 0;
}
