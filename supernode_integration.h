#ifndef SUPERNODE_INTEGRATION_H
#define SUPERNODE_INTEGRATION_H

#include <stdbool.h>
#include <stdint.h>
#include "system_integration.h"
#include "node_dna.h"
#include "autonomous_learning.h"

// Status codes for SuperNode operations
typedef enum {
    SUPERNODE_STATUS_OK = 0,
    SUPERNODE_STATUS_ERROR,
    SUPERNODE_STATUS_UNAVAILABLE,
    SUPERNODE_STATUS_TIMEOUT
} SuperNodeStatus;

// Structure to hold SuperNode connection state
typedef struct {
    bool initialized;
    char python_module_path[512];
    void* python_module;
    void* python_instance;
    char last_error[256];
    uint64_t requests_sent;
    uint64_t responses_received;
    uint64_t errors_encountered;
    uint64_t last_response_time;
    SystemIntegration* system_integration;
} SuperNodeConnection;

// Structure for dynamically loaded modules
typedef struct {
    char* path;
    void* handle; // Handle from dlopen
    // Add function pointers loaded from the module
    // Example: void (*module_process_func)(void*);
} DynamicModule;

// Structure for managing integration points
typedef struct {
    DynamicModule** modules;
    uint32_t module_count;
    uint32_t module_capacity;
    // Add other integration configurations
} SupernodeIntegrationManager;

// Initialize SuperNode connection
SuperNodeConnection* init_supernode_connection(SystemIntegration* integration, const char* module_path);

// Shutdown SuperNode connection
void destroy_supernode_connection(SuperNodeConnection* connection);

// Process data through SuperNode
SuperNodeStatus process_data_through_supernode(
    SuperNodeConnection* connection, 
    const float* data, 
    size_t data_size,
    float* result_buffer,
    size_t result_buffer_size
);

// Process text through SuperNode
SuperNodeStatus process_text_through_supernode(
    SuperNodeConnection* connection,
    const char* text,
    char* result_buffer,
    size_t result_buffer_size
);

// Get insights from SuperNode
SuperNodeStatus get_supernode_insights(
    SuperNodeConnection* connection,
    char* buffer,
    size_t buffer_size
);

// Transfer DNA to SuperNode
SuperNodeStatus transfer_dna_to_supernode(
    SuperNodeConnection* connection,
    const NodeDNA* dna
);

// Transfer knowledge gap to SuperNode
SuperNodeStatus transfer_knowledge_gap(
    SuperNodeConnection* connection,
    const char* topic,
    float priority
);

// Get quantum cube visualization data
SuperNodeStatus get_quantum_visualization_data(
    SuperNodeConnection* connection,
    float* vertices,
    size_t vertices_size,
    uint32_t* indices,
    size_t indices_size,
    uint32_t* vertex_count,
    uint32_t* index_count
);

// Check SuperNode connection status
bool is_supernode_available(SuperNodeConnection* connection);

// Get last error message
const char* get_supernode_error(SuperNodeConnection* connection);

// Function Prototypes
SupernodeIntegrationManager* init_supernode_integration(void);
void destroy_supernode_integration(SupernodeIntegrationManager* manager);

// Loads a dynamic module (e.g., a .so file)
bool load_integration_module(SupernodeIntegrationManager* manager, const char* module_path);

// Unloads a dynamic module
bool unload_integration_module(SupernodeIntegrationManager* manager, const char* module_path);

// Calls a specific function within a loaded module
void* call_module_function(SupernodeIntegrationManager* manager, const char* module_path, const char* function_name, void* arg);

// Gets the last error message
const char* supernode_integration_get_last_error(void);

#endif // SUPERNODE_INTEGRATION_H
