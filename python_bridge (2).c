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
    PYTHON_BRIDGE_STATUS_OK,
    PYTHON_BRIDGE_STATUS_NOT_INITIALIZED,
    PYTHON_BRIDGE_STATUS_ERROR
} PythonBridgeStatus;

// Python bridge structure
typedef struct PythonBridge PythonBridge;

// Function prototypes
/**
 * Initialize the Python bridge
 * @param integration System integration context
 * @param module_path Path to Python module
 * @param ingestion_layer Data ingestion layer for insights
 * @param supernode Conscious supernode for knowledge absorption
 * @return Pointer to initialized PythonBridge or NULL on failure
 */
PythonBridge* init_python_bridge(SystemIntegration* integration, const char* module_path,
                                DataIngestionLayer* ingestion_layer, ConsciousSuperNode* supernode);

/**
 * Check if Python module exists
 * @param bridge Python bridge
 * @param module_name Name of the Python module
 * @return true if module exists, false otherwise
 */
bool python_module_exists(PythonBridge* bridge, const char* module_name);

/**
 * Transfer node DNA to Python ecosystem
 * @param bridge Python bridge
 * @param node_id Node ID to transfer
 * @param python_module Python module name
 * @return Status code
 */
PythonBridgeStatus transfer_node_dna(PythonBridge* bridge, uint64_t node_id, const char* python_module);

/**
 * Evolve node DNA through Python module
 * @param bridge Python bridge
 * @param node_id Node ID to evolve
 * @return Status code
 */
PythonBridgeStatus evolve_dna_through_python(PythonBridge* bridge, uint64_t node_id);

/**
 * Process AI insights through Python
 * @param bridge Python bridge
 * @param insights_text Text of insights to process
 * @return Status code
 */
PythonBridgeStatus process_insights(PythonBridge* bridge, const char* insights_text);

/**
 * Create multidimensional perspective using Python module
 * @param bridge Python bridge
 * @param target_concept Concept for perspective
 * @param graph Memory graph to store perspective
 * @return Status code
 */
PythonBridgeStatus create_multidimensional_perspective(PythonBridge* bridge, const char* target_concept, MemoryGraph* graph);

/**
 * Mutate node DNA using Python
 * @param bridge Python bridge
 * @param node_id Node ID to mutate
 * @param mutation_rate Mutation rate (0.0 to 1.0)
 * @return Status code
 */
PythonBridgeStatus mutate_node_dna(PythonBridge* bridge, uint64_t node_id, float mutation_rate);

/**
 * Get the last error message
 * @param bridge Python bridge
 * @return Pointer to error message string
 */
const char* get_python_bridge_error(PythonBridge* bridge);

/**
 * Destroy the Python bridge
 * @param bridge Python bridge to destroy
 */
void destroy_python_bridge(PythonBridge* bridge);

#endif // PYTHON_BRIDGE_H