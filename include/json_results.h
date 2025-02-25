#ifndef JSON_RESULTS_H
#define JSON_RESULTS_H

#include <stdio.h>
#include <stdlib.h>

void save_results_to_json(const char *filename, const char *matrix_name, int num_threads, double execution_time);

#endif