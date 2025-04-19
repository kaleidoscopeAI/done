#ifndef SUPER_NODE_H
#define SUPER_NODE_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

// Forward declarations
typedef struct MemoryGraph MemoryGraph;
typedef struct SuperNodeScript SuperNodeScript;

// SuperNode types
typedef enum {
    SUPER_NODE_COORDINATOR,
    SUPER_NODE_MEMORY,
    SUPER_NODE_PROCESSING,
    SUPER_NODE_CUSTOM
} SuperNodeType;

// SuperNode structure
typedef struct SuperNode {
    uint64_t id;
    SuperNodeType type;
    char* name;
    uint64_t* controlled_node_ids; // Array of node IDs managed by this super node
    uint32_t controlled_node_count;
    uint32_t controlled_node_capacity;
    SuperNodeScript** scripts; // Array of scripts attached to this super node
    uint32_t script_count;
    uint32_t script_capacity;
    void* state_data; // SuperNode-specific state
    pthread_mutex_t lock; // For thread safety if needed
} SuperNode;

// Function Prototypes
SuperNode* create_super_node(uint64_t id, SuperNodeType type, const char* name);
SuperNode* create_coordinator_super_node(uint64_t id);
SuperNode* create_memory_super_node(uint64_t id);
SuperNode* create_processing_super_node(uint64_t id);

void destroy_super_node(SuperNode* sn);

bool super_node_add_controlled_node(SuperNode* sn, uint64_t node_id);
bool super_node_remove_controlled_node(SuperNode* sn, uint64_t node_id);

bool attach_script_to_super_node(SuperNode* sn, SuperNodeScript* script);
bool detach_script_from_super_node(SuperNode* sn, SuperNodeScript* script);

void process_super_node(SuperNode* sn, MemoryGraph* graph);

#endif // SUPER_NODE_H
