#ifndef PYTHON_BRIDGE_H
#define PYTHON_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "memory_graph.h"
#include "data_ingestion.h"
#include "conscious_supernode.h"

// Forward declaration
typedef struct SystemIntegration SystemIntegration;

// Status codes
typedef enum {
    PYTHON_BRIDGE_STATUS_OK = 0,
    PYTHON_BRIDGE_STATUS_NOT_INITIALIZED = 1,
    PYTHON_BRIDGE_STATUS_NULL_PTR = 2,
    PYTHON_BRIDGE_STATUS_MEMORY_ERROR = 3,
    PYTHON_BRIDGE_STATUS_MODULE_NOT_FOUND = 4,
    PYTHON_BRIDGE_STATUS_NODE_NOT_FOUND = 5
} PythonBridgeStatus;

// Python bridge structure
typedef struct PythonBridge PythonBridge;

// Function prototypes
PythonBridge* init_python_bridge(SystemIntegration* integration, const char* module_path,
                                DataIngestionLayer* ingestion_layer, ConsciousSuperNode* supernode);
PythonBridgeStatus python_module_exists(PythonBridge* bridge, const char* module_name);
PythonBridgeStatus transfer_node_dna(PythonBridge* bridge, uint64_t node_id, const char* python_module);
PythonBridgeStatus evolve_dna_through_python(PythonBridge* bridge, uint64_t node_id);
PythonBridgeStatus process_insights(PythonBridge* bridge, const char* insights_text);
PythonBridgeStatus create_multidimensional_perspective(PythonBridge* bridge, const char* target_concept, MemoryGraph* graph);
PythonBridgeStatus mutate_node_dna(PythonBridge* bridge, uint64_t node_id, float mutation_rate);
const char* get_python_bridge_error(PythonBridge* bridge);
void destroy_python_bridge(PythonBridge* bridge);

#endif // PYTHON_BRIDGE_H