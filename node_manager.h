#ifndef NODE_MANAGER_H
#define NODE_MANAGER_H

#include <stdbool.h>
#include <stdint.h>
#include "bridge_adapter.h"
#include "memory_graph.h"

// Error codes for NodeManager
#define NODE_MANAGER_SUCCESS 0
#define NODE_MANAGER_ERROR_INVALID_PARAMETER 1
#define NODE_MANAGER_ERROR_NODE_EXISTS 2
#define NODE_MANAGER_ERROR_NODE_NOT_FOUND 3
#define NODE_MANAGER_ERROR_BRIDGE_FAILED 4
#define NODE_MANAGER_ERROR_MEMORY_ALLOCATION 5

// Map of node types to integer constants
typedef enum {
    NODE_TYPE_STANDARD = 0,
    NODE_TYPE_CORE = 1,
    NODE_TYPE_MEMORY = 2,
    NODE_TYPE_PROCESSING = 3,
    NODE_TYPE_CUSTOM = 4
} NodeType;

// Structure to hold node properties
typedef struct {
    char* key;
    char* value;
} NodeProperty;

// Node structure
typedef struct {
    uint64_t id;
    NodeType type;
    NodeProperty* properties;
    uint32_t property_count;
    uint32_t property_capacity;
} ManagedNode;

// Node manager structure
typedef struct {
    ManagedNode* nodes;
    uint32_t node_count;
    uint32_t node_capacity;
    uint32_t next_node_id;
    int last_error;
    char error_message[256];
} NodeManager;

// Initialize node manager
NodeManager* init_node_manager(void);

// Create a node
int create_managed_node(NodeManager* manager, uint64_t node_id, NodeType node_type, const char* properties_json);

// Update a node
int update_managed_node(NodeManager* manager, uint64_t node_id, const char* properties_json);

// Get a node by ID
ManagedNode* get_managed_node(NodeManager* manager, uint64_t node_id);

// Delete a node
int delete_managed_node(NodeManager* manager, uint64_t node_id);

// Connect two nodes
int connect_managed_nodes(NodeManager* manager, uint64_t source_id, uint64_t target_id, double strength);

// Parse JSON properties
int parse_json_properties(ManagedNode* node, const char* properties_json);

// Destroy node manager
void destroy_node_manager(NodeManager* manager);

// Get last error message
const char* node_manager_get_last_error(NodeManager* manager);

#endif // NODE_MANAGER_H