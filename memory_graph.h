#ifndef MEMORY_GRAPH_H
#define MEMORY_GRAPH_H

#include <stdint.h>
#include <stdbool.h>

// Memory graph node structure
typedef struct {
    uint64_t id;
    void* data;
    float weight;
    uint32_t connections;
} MemoryNode;

// Memory graph structure
typedef struct {
    MemoryNode* nodes;
    uint32_t node_count;
    uint32_t max_nodes;
    void* graph_data;
} MemoryGraph;

// Function prototypes
MemoryGraph* create_memory_graph();
void free_memory_graph(MemoryGraph* graph);
int add_memory_node(MemoryGraph* graph, void* data, float weight);
int connect_memory_nodes(MemoryGraph* graph, uint64_t source_id, uint64_t target_id);
void* retrieve_memory_data(MemoryGraph* graph, uint64_t node_id);

#endif // MEMORY_GRAPH_H
