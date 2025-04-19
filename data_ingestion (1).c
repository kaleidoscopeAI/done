#ifndef DATA_INGESTION_H
#define DATA_INGESTION_H

#include <stdint.h>
#include <stdbool.h>
#include "web_crawler.h"
#include "memory_graph.h"
#include "bio_integration.h"
#include "conscious_supernode.h"

// Error codes
typedef enum {
    DATA_INGESTION_OK = 0,
    DATA_INGESTION_NULL_PTR = 1,
    DATA_INGESTION_INVALID_CONFIG = 2,
    DATA_INGESTION_MEMORY_ERROR = 3,
    DATA_INGESTION_CAPACITY_EXCEEDED = 4,
    DATA_INGESTION_CRAWLER_FAILED = 5,
    DATA_INGESTION_NOT_ACTIVE = 6
} DataIngestionStatus;

// Data source types
typedef enum {
    DATA_SOURCE_FILE,
    DATA_SOURCE_DATABASE,
    DATA_SOURCE_STREAM,
    DATA_SOURCE_API,
    DATA_SOURCE_WEBSOCKET,
    DATA_SOURCE_CUSTOM
} DataSourceType;

// Data ingestion configuration
typedef struct {
    DataSourceType source_type;
    char* source_path;
    char* credentials;
    uint32_t batch_size;
    uint32_t buffer_size;
    bool preprocessing_enabled;
    void* custom_config;
} DataIngestionConfig;

// Data ingestion layer
typedef struct {
    char** text_data;
    double** numerical_data;
    char** visual_data;
    uint64_t max_entries;
    uint64_t text_count;
    uint64_t numerical_count;
    uint64_t visual_count;
    DataIngestionConfig config;
    WebCrawler* web_crawler;
    MemoryGraph* memory_graph;
    BioIntegrationModule* bio_module;
    ConsciousSuperNode* supernode;
    uint64_t bytes_processed;
    uint32_t batches_processed;
    bool is_active;
    char last_error[256];
} DataIngestionLayer;

// Function prototypes
DataIngestionLayer* init_data_ingestion_layer(uint64_t max_entries, DataIngestionConfig* config,
                                             MemoryGraph* memory_graph, BioIntegrationModule* bio_module,
                                             ConsciousSuperNode* supernode);
DataIngestionStatus check_data_need(DataIngestionLayer* layer, const char* topic);
DataIngestionStatus ingest_text(DataIngestionLayer* layer, const char* text);
DataIngestionStatus ingest_disease_data(DataIngestionLayer* layer, const char* disease_data);
DataIngestionStatus add_to_ingestion_memory(DataIngestionLayer* layer, const char* memory_entry);
DataIngestionStatus ingest_numerical(DataIngestionLayer* layer, double* numbers, uint64_t count);
DataIngestionStatus ingest_visual(DataIngestionLayer* layer, const char* visual_input);
DataIngestionStatus start_ingestion(DataIngestionLayer* layer);
DataIngestionStatus stop_ingestion(DataIngestionLayer* layer);
void* get_next_batch(DataIngestionLayer* layer, DataIngestionStatus* status);
DataIngestionStatus process_data(DataIngestionLayer* layer, void* data, uint32_t size);
void get_ingested_data(DataIngestionLayer* layer);
void destroy_data_ingestion_layer(DataIngestionLayer* layer);
const char* data_ingestion_get_last_error(DataIngestionLayer* layer);

// Global instance
extern DataIngestionLayer* g_ingestion_layer;

#endif // DATA_INGESTION_H