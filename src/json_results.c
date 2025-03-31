#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/json_results.h"

void write_results_to_json(const char* filename, MatrixResult *results, int num_matrices) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    fprintf(fp, "[\n");
    for (int i = 0; i < num_matrices; i++) {
        fprintf(fp, "  {\n");

        // Nome matrice
        fprintf(fp, "    \"matrix_name\": \"%s\",\n", results[i].matrix_name);

        // Sezione "serial"
        fprintf(fp, "    \"serial\": {\n");
        fprintf(fp, "      \"time\": %.8f,\n", results[i].serial.time);
        fprintf(fp, "      \"flops\": %.8f\n", results[i].serial.flops);
        fprintf(fp, "    },\n");

        // Sezione "cuda"
        fprintf(fp, "    \"cuda\": {\n");
        fprintf(fp, "      \"time\": %.8f,\n", results[i].cuda.time);
        fprintf(fp, "      \"flops\": %.8f\n", results[i].cuda.flops);
        fprintf(fp, "    },\n");

        // Sezione "cuda_csr"
        fprintf(fp, "    \"cuda_csr\": [\n");
        for (int k = 0; k < results[i].num_cuda_csr; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"kernel_name\": \"%s\",\n", results[i].cuda_csr[k].kernel_name);
            fprintf(fp, "        \"time\": %.8f,\n", results[i].cuda_csr[k].time);
            fprintf(fp, "        \"gflops\": %.8f\n", results[i].cuda_csr[k].gflops);
            fprintf(fp, "      }%s\n", (k < results[i].num_cuda_csr - 1) ? "," : "");
        }
        fprintf(fp, "    ],\n");

        // Sezione "cuda_hll"
        fprintf(fp, "    \"cuda_hll\": [\n");
        for (int k = 0; k < results[i].num_cuda_hll; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"kernel_name\": \"%s\",\n", results[i].cuda_hll[k].kernel_name);
            fprintf(fp, "        \"time\": %.8f,\n", results[i].cuda_hll[k].time);
            fprintf(fp, "        \"gflops\": %.8f\n", results[i].cuda_hll[k].gflops);
            fprintf(fp, "      }%s\n", (k < results[i].num_cuda_hll - 1) ? "," : "");
        }
        fprintf(fp, "    ],\n");

        // Sezione "openmp_csr"
        fprintf(fp, "    \"openmp_csr\": [\n");
        for (int k = 0; k < results[i].num_openmp_csr; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"threads\": %d,\n", results[i].openmp_csr[k].threads);
            fprintf(fp, "        \"time\": %.8f,\n", results[i].openmp_csr[k].time);
            fprintf(fp, "        \"flops\": %.8f,\n", results[i].openmp_csr[k].flops);
            fprintf(fp, "        \"speedup\": %.8f,\n", results[i].openmp_csr[k].speedup);
            fprintf(fp, "        \"efficienza\": %.8f\n", results[i].openmp_csr[k].efficienza);
            fprintf(fp, "      }%s\n", (k < results[i].num_openmp_csr - 1) ? "," : "");
        }
        fprintf(fp, "    ],\n");

        // Sezione "openmp_hll"
        fprintf(fp, "    \"openmp_hll\": [\n");
        for (int k = 0; k < results[i].num_openmp_hll; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"threads\": %d,\n", results[i].openmp_hll[k].threads);
            fprintf(fp, "        \"time\": %.8f,\n", results[i].openmp_hll[k].time);
            fprintf(fp, "        \"flops\": %.8f,\n", results[i].openmp_hll[k].flops);
            fprintf(fp, "        \"speedup\": %.8f,\n", results[i].openmp_hll[k].speedup);
            fprintf(fp, "        \"efficienza\": %.8f\n", results[i].openmp_hll[k].efficienza);
            fprintf(fp, "      }%s\n", (k < results[i].num_openmp_hll - 1) ? "," : "");
        }
        fprintf(fp, "    ]\n");

        // Chiudiamo l'oggetto JSON
        fprintf(fp, "  }%s\n", (i < num_matrices - 1) ? "," : "");
    }
    fprintf(fp, "]\n");

    fclose(fp);
    printf("Risultati salvati in %s\n", filename);
}
