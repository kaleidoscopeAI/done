#ifndef MOCK_IMPLEMENTATION_H
#define MOCK_IMPLEMENTATION_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Mocking basic structures */
typedef struct {
    int id;
    int energy;
    int is_active;
} Node;

typedef struct MemoryNode {
    int id;
    char* description;
} MemoryNode;

typedef struct {
    MemoryNode** nodes;
    int count;
    int capacity;
} MemoryGraph;

typedef struct {
    int initialized;
    void* data;
} KaleidoscopeEngine;

typedef struct {
    char** buffers;
    int count;
    int capacity;
} DataIngestionLayer;

/* Mock functions for node_core.h */
Node* init_node(int id) {
    printf("[MOCK] Initializing Node with ID: %d\n", id);
    Node* node = (Node*)malloc(sizeof(Node));
    node->id = id;
    node->energy = 100;
    node->is_active = 1;
    return node;
}

void destroy_node(Node* node) {
    printf("[MOCK] Destroying Node\n");
    if (node) {
        free(node);
    }
}

/* Mock functions for memory_graph.h */
MemoryGraph* init_memory_graph(int capacity) {
    printf("[MOCK] Initializing Memory Graph with capacity: %d\n", capacity);
    MemoryGraph* graph = (MemoryGraph*)malloc(sizeof(MemoryGraph));
    graph->nodes = (MemoryNode**)malloc(sizeof(MemoryNode*) * capacity);
    graph->count = 0;
    graph->capacity = capacity;
    return graph;
}

void destroy_memory_graph(MemoryGraph* graph) {
    printf("[MOCK] Destroying Memory Graph\n");
    if (graph) {
        if (graph->nodes) {
            for (int i = 0; i < graph->count; i++) {
                if (graph->nodes[i]) {
                    if (graph->nodes[i]->description) free(graph->nodes[i]->description);
                    free(graph->nodes[i]);
                }
            }
            free(graph->nodes);
        }
        free(graph);
    }
}

MemoryNode* create_memory_node(const char* description, int value) {
    printf("[MOCK] Creating Memory Node: %s (value: %d)\n", description, value);
    MemoryNode* node = (MemoryNode*)malloc(sizeof(MemoryNode));
    node->id = rand() % 1000 + 1;
    node->description = strdup(description);
    return node;
}

void add_memory_node(MemoryGraph* graph, MemoryNode* node) {
    printf("[MOCK] Adding Memory Node to graph\n");
    if (graph && node && graph->count < graph->capacity) {
        graph->nodes[graph->count++] = node;
    }
}

/* Mock functions for kaleidoscope_engine.h */
KaleidoscopeEngine* init_kaleidoscope_engine() {
    printf("[MOCK] Initializing Kaleidoscope Engine\n");
    KaleidoscopeEngine* engine = (KaleidoscopeEngine*)malloc(sizeof(KaleidoscopeEngine));
    engine->initialized = 1;
    engine->data = NULL;
    return engine;
}

void destroy_kaleidoscope_engine(KaleidoscopeEngine* engine) {
    printf("[MOCK] Destroying Kaleidoscope Engine\n");
    if (engine) {
        free(engine);
    }
}

void simulate_compound_interaction(KaleidoscopeEngine* engine, const char* sample_data) {
    printf("[MOCK] Simulating compound interaction with data: %s\n", sample_data);
}

/* Mock functions for data_ingestion.h */
DataIngestionLayer* init_data_ingestion_layer(int buffer_size) {
    printf("[MOCK] Initializing Data Ingestion Layer with buffer size: %d\n", buffer_size);
    DataIngestionLayer* layer = (DataIngestionLayer*)malloc(sizeof(DataIngestionLayer));
    layer->buffers = NULL;
    layer->count = 0;
    layer->capacity = buffer_size;
    return layer;
}

void destroy_data_ingestion_layer(DataIngestionLayer* layer) {
    printf("[MOCK] Destroying Data Ingestion Layer\n");
    if (layer) {
        if (layer->buffers) {
            for (int i = 0; i < layer->count; i++) {
                if (layer->buffers[i]) free(layer->buffers[i]);
            }
            free(layer->buffers);
        }
        free(layer);
    }
}

/* Mock functions for websocket_server.h */
int init_websocket_server() {
    printf("[MOCK] Initializing WebSocket Server\n");
    return 0;
}

void service_websocket() {
    // Just a stub, no output to avoid spam
}

void destroy_websocket_server() {
    printf("[MOCK] Destroying WebSocket Server\n");
}

#endif /* MOCK_IMPLEMENTATION_H */