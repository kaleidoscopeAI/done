#ifndef BRIDGE_ADAPTER_H
#define BRIDGE_ADAPTER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define BRIDGE_ERROR_NONE 0
#define BRIDGE_ERROR_NOT_INITIALIZED 1
#define BRIDGE_ERROR_MEMORY_ALLOCATION 2
#define BRIDGE_ERROR_INVALID_PARAMETER 3
#define BRIDGE_ERROR_NODE_NOT_FOUND 4
#define BRIDGE_ERROR_OPERATION_FAILED 5

// Callback function type for event notifications from C to Python
typedef void (*EventCallback)(const char* event_name, const char* data);

/**
 * Initialize the system
 * 
 * @return 0 on success, non-zero on failure
 */
int initialize_system(void);

/**
 * Shutdown the system
 * 
 * @return 0 on success, non-zero on failure
 */
int shutdown_system(void);

/**
 * Configure the system with the given configuration string
 * 
 * @param config_str Configuration string in format "key1=value1;key2=value2"
 * @return 0 on success, non-zero on failure
 */
int configure_system(const char* config_str);

/**
 * Create a node in the memory graph
 * 
 * @param node_id Node ID to use
 * @param node_type Node type (0=standard, 1=core, etc.)
 * @return Node pointer or 0 on failure
 */
uint64_t bridge_create_node(uint64_t node_id, int node_type);

/**
 * Connect two nodes in the memory graph
 * 
 * @param node1 First node pointer
 * @param node2 Second node pointer
 * @param strength Connection strength
 * @return 1 on success, 0 on failure
 */
int bridge_connect_nodes(uint64_t node1, uint64_t node2, double strength);

/**
 * Update the memory graph node with new data
 * 
 * @param node_id Node ID to update
 * @param data_json JSON string containing node data
 * @return 1 on success, 0 on failure
 */
int bridge_update_memory_graph(uint64_t node_id, const char* data_json);

/**
 * Get updated nodes from the memory graph
 * 
 * @return JSON string containing updated nodes data (must be freed by Python)
 */
const char* bridge_get_updated_nodes(void);

/**
 * Remove a node from the memory graph
 * 
 * @param node_id Node ID to remove
 * @return 1 on success, 0 on failure
 */
int bridge_remove_node(uint64_t node_id);

/**
 * Initialize the task manager
 * 
 * @param max_tasks Maximum number of tasks
 * @return Task manager pointer or NULL on failure
 */
void* init_task_manager(int max_tasks);

/**
 * Add a task to the task manager
 * 
 * @param manager Task manager pointer
 * @param task_id Task ID
 * @param priority Task priority (0-255)
 * @param assigned_node_id Node ID to assign task to (0 for unassigned)
 * @param task_data Task data
 * @return 1 on success, 0 on failure
 */
int add_task(void* manager, int task_id, int priority, uint64_t assigned_node_id, const char* task_data);

/**
 * Get the next task from the task manager
 * 
 * @param manager Task manager pointer
 * @return JSON string with task data or NULL if no tasks (must be freed by Python)
 */
const char* get_next_task(void* manager);

/**
 * Complete a task in the task manager
 * 
 * @param manager Task manager pointer
 * @param task_id Task ID
 * @return 1 on success, 0 on failure
 */
int complete_task(void* manager, int task_id);

/**
 * Register a callback for events
 * 
 * @param event_name Event name to register for (or "*" for all events)
 * @param callback Callback function
 * @return 1 on success, 0 on failure
 */
int register_callback(const char* event_name, EventCallback callback);

/**
 * Get last error code
 * 
 * @return Error code
 */
int bridge_get_last_error(void);

/**
 * Get last error message
 * 
 * @return Error message string
 */
const char* bridge_get_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_ADAPTER_H