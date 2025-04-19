#ifndef DATA_INGESTION_H
#define DATA_INGESTION_H

#include <stdint.h>
#include "binary_converter.h"

// Data ingestion structure
typedef struct {
    char** binary_entries;
    uint64_t max_entries;
    uint64_t current_count;
} DataIngestionLayer;

// Function declarations
DataIngestionLayer* init_data_ingestion_layer(uint64_t max_entries);
void ingest_data(DataIngestionLayer* layer, const char* text);
void free_ingestion_layer(DataIngestionLayer* layer);

#endif // DATA_INGESTION_H

