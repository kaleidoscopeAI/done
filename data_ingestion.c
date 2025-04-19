#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "data_ingestion.h"
#include "web_crawler.h"
#include "bridge_adapter.h"

// Global instance
DataIngestionLayer* g_ingestion_layer = NULL;
WebCrawler* g_crawler = NULL;

// Error handling
static void set_ingestion_error(DataIngestionLayer* layer, const char* format, ...) {
    if (!layer) return;
    va_list args;
    va_start(args, format);
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    snprintf(layer->last_error, sizeof(layer->last_error), "[%s] %s", timestamp, format);
    vsnprintf(layer->last_error, sizeof(layer->last_error), layer->last_error, args);
    va_end(args);
    fprintf(stderr, "[Data Ingestion Error] %s\n", layer->last_error);
}

const char* data_ingestion_get_last_error(DataIngestionLayer* layer) {
    return layer ? layer->last_error : "Layer not initialized";
}

DataIngestionLayer* init_data_ingestion_layer(uint64_t max_entries, DataIngestionConfig* config,
                                             MemoryGraph* memory_graph, BioIntegrationModule* bio_module,
                                             ConsciousSuperNode* supernode) {
    if (max_entries == 0) {
        fprintf(stderr, "[Data Ingestion Error] Invalid max_entries value\n");
        return NULL;
    }

    DataIngestionLayer* layer = (DataIngestionLayer*)calloc(1, sizeof(DataIngestionLayer));
    if (!layer) {
        fprintf(stderr, "[Data Ingestion Error] Failed to allocate memory for Data Ingestion Layer\n");
        return NULL;
    }

    layer->text_data = (char**)calloc(max_entries, sizeof(char*));
    layer->numerical_data = (double**)calloc(max_entries, sizeof(double*));
    layer->visual_data = (char**)calloc(max_entries, sizeof(char*));
    
    if (!layer->text_data || !layer->numerical_data || !layer->visual_data) {
        set_ingestion_error(layer, "Failed to allocate memory for data arrays");
        free(layer->text_data);
        free(layer->numerical_data);
        free(layer->visual_data);
        free(layer);
        return NULL;
    }
    
    layer->max_entries = max_entries;
    layer->text_count = 0;
    layer->numerical_count = 0;
    layer->visual_count = 0;
    layer->memory_graph = memory_graph;
    layer->bio_module = bio_module;
    layer->supernode = supernode;
    layer->is_active = false;
    layer->bytes_processed = 0;
    layer->batches_processed = 0;
    time_t now = time(NULL);
    strftime(layer->last_error, sizeof(layer->last_error), "[%Y-%m-%d %H:%M:%S] No error", localtime(&now));

    if (config) {
        if (config->batch_size == 0 || config->buffer_size == 0) {
            set_ingestion_error(layer, "Invalid batch or buffer size in config");
            destroy_data_ingestion_layer(layer);
            return NULL;
        }
        layer->config = *config;
    } else {
        layer->config.source_type = DATA_SOURCE_CUSTOM;
        layer->config.batch_size = 100;
        layer->config.buffer_size = 4096;
        layer->config.preprocessing_enabled = true;
    }

    layer->web_crawler = init_web_crawler(layer);
    g_crawler = layer->web_crawler;
    
    if (!layer->web_crawler) {
        set_ingestion_error(layer, "Failed to initialize web crawler");
        destroy_data_ingestion_layer(layer);
        return NULL;
    }

    CrawlerConfig crawler_config = {
        .max_pages_per_domain = 5,
        .crawl_delay_ms = 500,
        .timeout_seconds = 30,
        .max_results = 50,
        .max_crawl_depth = 3,
        .respect_robots_txt = true,
        .follow_redirects = true
    };
    strcpy(crawler_config.user_agent, "KaleidoscopeAI-Bot/1.0 (+https://example.com/bot)");
    if (configure_web_crawler(layer->web_crawler, &crawler_config) != 0) {
        set_ingestion_error(layer, "Failed to configure web crawler");
        destroy_data_ingestion_layer(layer);
        return NULL;
    }

    g_ingestion_layer = layer;
    printf("Data Ingestion Layer initialized with capacity for %lu entries per type\n", max_entries);
    return layer;
}

DataIngestionStatus check_data_need(DataIngestionLayer* layer, const char* topic) {
    if (!layer || !topic || !g_crawler) {
        set_ingestion_error(layer, "Invalid arguments or uninitialized crawler");
        return DATA_INGESTION_NULL_PTR;
    }
    
    bool found = false;
    for (uint64_t i = 0; i < layer->text_count; i++) {
        if (layer->text_data[i] && strstr(layer->text_data[i], topic)) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        printf("No data found for topic '%s', starting autonomous crawl\n", topic);
        if (!autonomous_data_crawl(g_crawler, topic, false)) {
            set_ingestion_error(layer, "Autonomous crawl failed for topic: %s", topic);
            return DATA_INGESTION_CRAWLER_FAILED;
        }
        if (ingest_text(layer, topic) != DATA_INGESTION_OK) {
            return DATA_INGESTION_MEMORY_ERROR;
        }
        if (layer->supernode) {
            absorb_knowledge(layer->supernode, topic);
        }
    }
    
    return DATA_INGESTION_OK;
}

DataIngestionStatus ingest_text(DataIngestionLayer* layer, const char* text) {
    if (!layer || !text) {
        set_ingestion_error(layer, "Invalid arguments");
        return DATA_INGESTION_NULL_PTR;
    }
    if (layer->text_count >= layer->max_entries) {
        set_ingestion_error(layer, "Text ingestion capacity reached");
        return DATA_INGESTION_CAPACITY_EXCEEDED;
    }

    char* text_copy = strdup(text);
    if (!text_copy) {
        set_ingestion_error(layer, "Failed to allocate memory for text data");
        return DATA_INGESTION_MEMORY_ERROR;
    }

    layer->text_data[layer->text_count++] = text_copy;
    layer->bytes_processed += strlen(text);
    
    if (layer->memory_graph) {
        char json_data[4096];
        snprintf(json_data, sizeof(json_data), "{\"type\":\"text\",\"data\":\"%s\"}", text);
        MemoryNode* node = create_memory_node(json_data, 1.0);
        if (!node || add_memory_node(layer->memory_graph, node) != 0) {
            if (node) destroy_memory_node(node);
            set_ingestion_error(layer, "Failed to add text to memory graph");
            free(text_copy);
            layer->text_data[--layer->text_count] = NULL;
            return DATA_INGESTION_MEMORY_ERROR;
        }
    }

    if (layer->supernode) {
        absorb_knowledge(layer->supernode, text);
    }

    printf("Ingested text data: %.50s...\n", text);
    return DATA_INGESTION_OK;
}

DataIngestionStatus ingest_disease_data(DataIngestionLayer* layer, const char* disease_data) {
    if (!layer || !disease_data) {
        set_ingestion_error(layer, "Invalid arguments");
        return DATA_INGESTION_NULL_PTR;
    }
    if (layer->text_count >= layer->max_entries) {
        set_ingestion_error(layer, "Disease data ingestion capacity reached");
        return DATA_INGESTION_CAPACITY_EXCEEDED;
    }
    
    char* data_copy = strdup(disease_data);
    if (!data_copy) {
        set_ingestion_error(layer, "Failed to allocate memory for disease data");
        return DATA_INGESTION_MEMORY_ERROR;
    }

    layer->text_data[layer->text_count++] = data_copy;
    layer->bytes_processed += strlen(disease_data);
    
    if (layer->bio_module) {
        BioData bio_data = {
            .sequence_id = strdup("disease"),
            .sequence_data = data_copy,
            .metadata = NULL
        };
        if (!bio_data.sequence_id || process_bio_data(layer->bio_module, &bio_data, NULL) != 0) {
            set_ingestion_error(layer, "Failed to process bio data");
            free(bio_data.sequence_id);
            free(data_copy);
            layer->text_data[--layer->text_count] = NULL;
            return DATA_INGESTION_MEMORY_ERROR;
        }
        free(bio_data.sequence_id);
    }

    char disease_name[256] = "";
    if (sscanf(disease_data, "Disease: %255[^\n]", disease_name) == 1 && g_crawler && disease_name[0]) {
        char search_query[512];
        snprintf(search_query, sizeof(search_query), "%s symptoms treatment research", disease_name);
        if (!autonomous_data_crawl(g_crawler, search_query, true)) {
            set_ingestion_error(layer, "Failed to crawl disease data for: %s", disease_name);
        }
    }

    printf("Ingested disease data: %.50s...\n", disease_data);
    return DATA_INGESTION_OK;
}

DataIngestionStatus add_to_ingestion_memory(DataIngestionLayer* layer, const char* memory_entry) {
    if (!layer || !memory_entry) {
        set_ingestion_error(layer, "Invalid arguments");
        return DATA_INGESTION_NULL_PTR;
    }
    if (layer->text_count >= layer->max_entries) {
        set_ingestion_error(layer, "Memory ingestion capacity reached");
        return DATA_INGESTION_CAPACITY_EXCEEDED;
    }
    
    char* entry_copy = strdup(memory_entry);
    if (!entry_copy) {
        set_ingestion_error(layer, "Failed to allocate memory for memory entry");
        return DATA_INGESTION_MEMORY_ERROR;
    }

    layer->text_data[layer->text_count++] = entry_copy;
    layer->bytes_processed += strlen(memory_entry);
    
    if (layer->memory_graph) {
        char json_data[4096];
        snprintf(json_data, sizeof(json_data), "{\"type\":\"memory\",\"data\":\"%s\"}", memory_entry);
        MemoryNode* node = create_memory_node(json_data, 1.0);
        if (!node || add_memory_node(layer->memory_graph, node) != 0) {
            if (node) destroy_memory_node(node);
            set_ingestion_error(layer, "Failed to add memory entry to memory graph");
            free(entry_copy);
            layer->text_data[--layer->text_count] = NULL;
            return DATA_INGESTION_MEMORY_ERROR;
        }
    }

    printf("Added to ingestion memory: %.50s...\n", memory_entry);
    return DATA_INGESTION_OK;
}

DataIngestionStatus ingest_numerical(DataIngestionLayer* layer, double* numbers, uint64_t count) {
    if (!layer || !numbers || count == 0) {
        set_ingestion_error(layer, "Invalid arguments");
        return DATA_INGESTION_NULL_PTR;
    }
    if (layer->numerical_count >= layer->max_entries) {
        set_ingestion_error(layer, "Numerical ingestion capacity reached");
        return DATA_INGESTION_CAPACITY_EXCEEDED;
    }

    double* entry = (double*)malloc(sizeof(double) * count);
    if (!entry) {
        set_ingestion_error(layer, "Failed to allocate memory for numerical data");
        return DATA_INGESTION_MEMORY_ERROR;
    }

    memcpy(entry, numbers, sizeof(double) * count);
    layer->numerical_data[layer->numerical_count++] = entry;
    layer->bytes_processed += sizeof(double) * count;

    if (layer->memory_graph) {
        char json_data[4096];
        snprintf(json_data, sizeof(json_data), "{\"type\":\"numerical\",\"count\":%lu}", count);
        MemoryNode* node = create_memory_node(json_data, 1.0);
        if (!node || add_memory_node(layer->memory_graph, node) != 0) {
            if (node) destroy_memory_node(node);
            set_ingestion_error(layer, "Failed to add numerical data to memory graph");
            free(entry);
            layer->numerical_data[--layer->numerical_count] = NULL;
            return DATA_INGESTION_MEMORY_ERROR;
        }
    }

    printf("Ingested numerical data: [");
    for (uint64_t i = 0; i < count && i < 5; i++) {
        printf("%lf%s", numbers[i], (i < count - 1 && i < 4) ? ", " : "");
    }
    if (count > 5) printf("...");
    printf("]\n");
    return DATA_INGESTION_OK;
}

DataIngestionStatus ingest_visual(DataIngestionLayer* layer, const char* visual_input) {
    if (!layer || !visual_input) {
        set_ingestion_error(layer, "Invalid arguments");
        return DATA_INGESTION_NULL_PTR;
    }
    if (layer->visual_count >= layer->max_entries) {
        set_ingestion_error(layer, "Visual ingestion capacity reached");
        return DATA_INGESTION_CAPACITY_EXCEEDED;
    }

    char* input_copy = strdup(visual_input);
    if (!input_copy) {
        set_ingestion_error(layer, "Failed to allocate memory for visual data");
        return DATA_INGESTION_MEMORY_ERROR;
    }

    layer->visual_data[layer->visual_count++] = input_copy;
    layer->bytes_processed += strlen(visual_input);

    if (layer->memory_graph) {
        char json_data[4096];
        snprintf(json_data, sizeof(json_data), "{\"type\":\"visual\",\"data\":\"%s\"}", visual_input);
        MemoryNode* node = create_memory_node(json_data, 1.0);
        if (!node || add_memory_node(layer->memory_graph, node) != 0) {
            if (node) destroy_memory_node(node);
            set_ingestion_error(layer, "Failed to add visual data to memory graph");
            free(input_copy);
            layer->visual_data[--layer->visual_count] = NULL;
            return DATA_INGESTION_MEMORY_ERROR;
        }
    }

    printf("Ingested visual data: %.50s...\n", visual_input);
    return DATA_INGESTION_OK;
}

DataIngestionStatus start_ingestion(DataIngestionLayer* layer) {
    if (!layer) {
        set_ingestion_error(NULL, "Layer is uninitialized");
        return DATA_INGESTION_NULL_PTR;
    }

    if (layer->is_active) {
        set_ingestion_error(layer, "Ingestion already active");
        return DATA_INGESTION_NOT_ACTIVE;
    }

    layer->is_active = true;
    layer->batches_processed = 0;
    printf("Data ingestion started\n");
    return DATA_INGESTION_OK;
}

DataIngestionStatus stop_ingestion(DataIngestionLayer* layer) {
    if (!layer) {
        set_ingestion_error(NULL, "Layer is uninitialized");
        return DATA_INGESTION_NULL_PTR;
    }

    if (!layer->is_active) {
        set_ingestion_error(layer, "Ingestion not active");
        return DATA_INGESTION_NOT_ACTIVE;
    }

    layer->is_active = false;
    printf("Data ingestion stopped\n");
    return DATA_INGESTION_OK;
}

void* get_next_batch(DataIngestionLayer* layer, DataIngestionStatus* status) {
    if (!layer || !status) {
        set_ingestion_error(layer, "Invalid arguments");
        if (status) *status = DATA_INGESTION_NULL_PTR;
        return NULL;
    }
    if (!layer->is_active) {
        set_ingestion_error(layer, "Layer is not active");
        *status = DATA_INGESTION_NOT_ACTIVE;
        return NULL;
    }

    if (layer->text_count == 0 && layer->numerical_count == 0 && layer->visual_count == 0) {
        *status = DATA_INGESTION_OK;
        return NULL;
    }

    char* batch_data = (char*)calloc(layer->config.batch_size, sizeof(char));
    if (!batch_data) {
        set_ingestion_error(layer, "Failed to allocate memory for batch data");
        *status = DATA_INGESTION_MEMORY_ERROR;
        return NULL;
    }

    size_t offset = 0;
    // Process text data
    for (uint64_t i = layer->text_count > 0 ? layer->text_count - 1 : 0; i < layer->text_count && offset < layer->config.batch_size - 512; i++) {
        if (layer->text_data[i]) {
            offset += snprintf(batch_data + offset, layer->config.batch_size - offset, "{\"type\":\"text\",\"data\":\"%s\"}", layer->text_data[i]);
        }
    }
    // Process numerical data
    for (uint64_t i = layer->numerical_count > 0 ? layer->numerical_count - 1 : 0; i < layer->numerical_count && offset < layer->config.batch_size - 512; i++) {
        if (layer->numerical_data[i]) {
            offset += snprintf(batch_data + offset, layer->config.batch_size - offset, "{\"type\":\"numerical\",\"data\":[...]}");
        }
    }
    // Process visual data
    for (uint64_t i = layer->visual_count > 0 ? layer->visual_count - 1 : 0; i < layer->visual_count && offset < layer->config.batch_size - 512; i++) {
        if (layer->visual_data[i]) {
            offset += snprintf(batch_data + offset, layer->config.batch_size - offset, "{\"type\":\"visual\",\"data\":\"%s\"}", layer->visual_data[i]);
        }
    }

    layer->batches_processed++;
    *status = DATA_INGESTION_OK;
    return batch_data;
}

DataIngestionStatus process_data(DataIngestionLayer* layer, void* data, uint32_t size) {
    if (!layer || !data || size == 0) {
        set_ingestion_error(layer, "Invalid arguments");
        return DATA_INGESTION_NULL_PTR;
    }
    if (!layer->is_active) {
        set_ingestion_error(layer, "Ingestion not active");
        return DATA_INGESTION_NOT_ACTIVE;
    }

    if (layer->config.preprocessing_enabled) {
        DataIngestionStatus status = ingest_text(layer, (const char*)data);
        if (status != DATA_INGESTION_OK) {
            return status;
        }
    }

    layer->bytes_processed += size;
    layer->batches_processed++;
    return DATA_INGESTION_OK;
}

void get_ingested_data(DataIngestionLayer* layer) {
    if (!layer) {
        printf("Error: Layer is uninitialized\n");
        return;
    }
    
    printf("Data Ingestion Layer Status:\n");
    printf("- Text entries: %lu / %lu\n", layer->text_count, layer->max_entries);
    printf("- Numerical entries: %lu / %lu\n", layer->numerical_count, layer->max_entries);
    printf("- Visual entries: %lu / %lu\n", layer->visual_count, layer->max_entries);
    printf("- Bytes processed: %lu\n", layer->bytes_processed);
    printf("- Batches processed: %u\n", layer->batches_processed);
    printf("- Active: %s\n", layer->is_active ? "Yes" : "No");
    printf("- Last error: %s\n", layer->last_error);
}

void destroy_data_ingestion_layer(DataIngestionLayer* layer) {
    if (!layer) return;

    for (uint64_t i = 0; i < layer->text_count; i++) {
        free(layer->text_data[i]);
    }
    for (uint64_t i = 0; i < layer->numerical_count; i++) {
        free(layer->numerical_data[i]);
    }
    for (uint64_t i = 0; i < layer->visual_count; i++) {
        free(layer->visual_data[i]);
    }

    free(layer->text_data);
    free(layer->numerical_data);
    free(layer->visual_data);
    
    if (g_crawler) {
        destroy_web_crawler(g_crawler);
        g_crawler = NULL;
    }
    
    free(layer);
    g_ingestion_layer = NULL;
    printf("Data Ingestion Layer destroyed\n");
}