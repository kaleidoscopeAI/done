#ifndef NODE_CORE_H
#define NODE_CORE_H

#include <stdint.h>
#include <stdbool.h>

// Basic node structure
typedef struct {
    uint64_t id;             // Unique node identifier
    char* name;              // Node name
    uint32_t type;           // Node type classifier
    void* data;              // Pointer to node-specific data
    uint32_t connection_count;  // Number of connections to other nodes
    struct Node** connections;  // Array of connections to other nodes
    float activation_level;     // Current activation level (0.0 to 1.0)
    bool active;                // Whether the node is currently active
    void (*process_func)(struct Node*);  // Function pointer for node processing
} Node;

/**
 * Node state structure for websocket transmission
 */
typedef struct {
    double position[3];  // x, y, z position in 3D space
    double energy;       // energy level of the node
    char type[32];       // node type identifier
} NodeState;

/**
 * Gets the current state of all nodes in the simulation
 * 
 * @param count Pointer to int where the number of nodes will be stored
 * @return Array of NodeState structures (caller must free)
 */
NodeState* get_node_states(int *count);

/**
 * Adds a new node to the simulation with the specified type
 * 
 * @param type The type of node to add
 * @return 0 on success, non-zero on failure
 */
int add_node_to_simulation(const char *type);

// Function prototypes
Node* create_node(uint64_t id, const char* name, uint32_t type);
void connect_nodes(Node* source, Node* target);
void activate_node(Node* node, float level);
void process_node(Node* node);
void free_node(Node* node);

// Structure for node core
typedef struct {
    void* data;
    // Other fields as needed
} NodeCore;

// Function to initialize the node core
NodeCore* init_node_core(void);

// Function to process a node request
int process_node_request(NodeCore* core, const char* request);

// Function to clean up resources
void free_node_core(NodeCore* core);

#endif // NODE_CORE_H

