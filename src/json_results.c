#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/json_results.h"

void write_results_to_json(const char* filename, MatrixResult *results, int num_matrices) {
    FILE *fp = fopen(filename, "w");
    if(!fp){
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    fprintf(fp, "[\n");
    for (int i = 0; i < num_matrices; i++) {
        fprintf(fp, "  {\n");

        // Nome matrice
        fprintf(fp, "    \"matrix_name\": \"%s\",\n", results[i].matrix_name);

        // Sezione "serial" (usiamo "time" e "flops")
        fprintf(fp, "    \"serial\": {\n");
        fprintf(fp, "      \"time\": %.8f,\n",  results[i].serial.time);
        fprintf(fp, "      \"flops\": %.8f\n", results[i].serial.flops);
        fprintf(fp, "    },\n");

        // Sezione "cuda" (solo 1 record se lo usi così)
        fprintf(fp, "    \"cuda\": {\n");
        fprintf(fp, "      \"time\": %.8f,\n",  results[i].cuda.time);
        fprintf(fp, "      \"flops\": %.8f\n", results[i].cuda.flops);
        fprintf(fp, "    },\n");

        // Sezione "cuda_csr" (array con i diversi kernel)
        fprintf(fp, "    \"cuda_csr\": [\n");
        for(int k = 0; k < results[i].num_cuda_csr; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"kernel_name\": \"%s\",\n", results[i].cuda_csr[k].kernel_name);
            fprintf(fp, "        \"time\": %.8f,\n",         results[i].cuda_csr[k].time);
            fprintf(fp, "        \"gflops\": %.8f\n",        results[i].cuda_csr[k].gflops);
            fprintf(fp, "      }");
            if(k < results[i].num_cuda_csr - 1) {
                fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "    ],\n");

        // Sezione "cuda_hll"
        fprintf(fp, "    \"cuda_hll\": [\n");
        for(int k = 0; k < results[i].num_cuda_hll; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"kernel_name\": \"%s\",\n", results[i].cuda_hll[k].kernel_name);
            fprintf(fp, "        \"time\": %.8f,\n",          results[i].cuda_hll[k].time);
            fprintf(fp, "        \"gflops\": %.8f\n",         results[i].cuda_hll[k].gflops);
            fprintf(fp, "      }");
            if(k < results[i].num_cuda_hll - 1) {
                fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "    ],\n");

        // Sezione "openmp"
        fprintf(fp, "    \"openmp\": [\n");
        for(int k = 0; k < results[i].num_openmp; k++) {
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"threads\": %d,\n",   results[i].openmp_results[k].threads);
            fprintf(fp, "        \"time\": %.8f,\n",    results[i].openmp_results[k].time);
            fprintf(fp, "        \"flops\": %.8f\n",    results[i].openmp_results[k].flops);
            fprintf(fp, "      }");
            if(k < results[i].num_openmp - 1) {
                fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "    ]\n");

        // Chiudiamo l'oggetto
        fprintf(fp, "  }");

        // Se non è l'ultimo, appendi la virgola
        if(i < num_matrices - 1) {
            fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "]\n");

    fclose(fp);
    printf("Risultati salvati in %s\n", filename);
}
