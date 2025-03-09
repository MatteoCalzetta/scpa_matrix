#include <stdio.h>
#include <stdlib.h>
#include "../include/json_results.h"

void write_results_to_json(const char *filename, MatrixResult *results, int num_matrices) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Errore: impossibile aprire %s per scrittura\n", filename);
        return;
    }

    fprintf(file, "{\n  \"matrices\": [\n");

    for (int i = 0; i < num_matrices; i++) {
        fprintf(file, "    {\n      \"name\": \"%s\",\n      \"executions\": {\n", results[i].matrix_name);

        // Sezione seriale
        fprintf(file, "        \"seriale\": {\n          \"time\": %.10f\n        },\n", results[i].serial.time);

        // Sezione CUDA
        fprintf(file, "        \"cuda\": {\n          \"time\": %.10f,\n          \"flops\": %.10f\n        },\n",
                results[i].cuda.time, results[i].cuda.flops);

        // Sezione OpenMP
        fprintf(file, "        \"openmp\": [\n");
        for (int j = 0; j < results[i].num_openmp; j++) {
            fprintf(file, "          {\n            \"threads\": %d,\n            \"time\": %.10f,\n            \"flops\": %.10f\n          }",
                    results[i].openmp_results[j].threads,
                    results[i].openmp_results[j].time,
                    results[i].openmp_results[j].flops);
            if (j < results[i].num_openmp - 1) {
                fprintf(file, ",");
            }
            fprintf(file, "\n");
        }
        fprintf(file, "        ]\n");

        fprintf(file, "      }\n    }");

        if (i < num_matrices - 1) {
            fprintf(file, ",");
        }
        fprintf(file, "\n");
    }

    fprintf(file, "  ]\n}\n");
    fclose(file);
    printf("Risultati salvati in %s\n", filename);
}