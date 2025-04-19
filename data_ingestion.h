#ifndef DATA_INGESTION_H
#define DATA_INGESTION_H

#include <stdint.h>
#include <stdbool.h>

// Forward declare WebCrawler struct
typedef struct WebCrawler WebCrawler;

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
    void* data;
    DataIngestionConfig config;
    uint64_t bytes_processed;
    uint32_t batches_processed;
    bool is_active;
    WebCrawler* web_crawler; // Add web_crawler field
} DataIngestionLayer;

// Function prototypes
DataIngestionLayer* init_data_ingestion(DataIngestionConfig* config);
int start_ingestion(DataIngestionLayer* layer);
int stop_ingestion(DataIngestionLayer* layer);
void* get_next_batch(DataIngestionLayer* layer);
int process_data(DataIngestionLayer* layer, void* data, uint32_t size);
void free_data_ingestion(DataIngestionLayer* layer);

// Global instance
extern DataIngestionLayer* g_ingestion_layer;

#endif // DATA_INGESTION_H

