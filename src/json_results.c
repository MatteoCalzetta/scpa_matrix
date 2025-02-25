#include "../include/json_results.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_FILE "results.json"

void save_results_to_json(const char *filename, const char *matrix_name, int num_threads, double execution_time) {
    FILE *file = fopen(JSON_FILE, "a");  // Apriamo in modalità append
    if (!file) {
        printf("❌ Errore: impossibile aprire %s\n", JSON_FILE);
        return;
    }

    // Scriviamo i risultati nel file JSON
    fprintf(file, "{ \"matrix\": \"%s\", \"threads\": %d, \"time\": %f },\n", matrix_name, num_threads, execution_time);

    fclose(file);
}