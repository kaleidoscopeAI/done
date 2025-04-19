#ifndef QUANTUM_CUBE_VISUALIZER_H
#define QUANTUM_CUBE_VISUALIZER_H

#include <stdint.h>
#include <stdbool.h>

// Quantum node representation for visualization
typedef struct {
    float x, y, z;           // 3D position
    float size;              // Node size
    float r, g, b;           // RGB color
    float energy;            // Energy level
    int type;                // Node type
    char label[64];          // Node label
    bool active;             // Active state
} QuantumNode;

// Connection between quantum nodes
typedef struct {
    int source;              // Source node index
    int target;              // Target node index
    float strength;          // Connection strength
    float r, g, b;           // RGB color
} QuantumConnection;

// State of the quantum cube
typedef struct {
    QuantumNode* nodes;              // Array of nodes
    QuantumConnection* connections;  // Array of connections
    int node_count;                  // Number of nodes
    int connection_count;            // Number of connections
    int max_nodes;                   // Maximum number of nodes
    int max_connections;             // Maximum number of connections
    float rotation_x;                // X rotation angle
    float rotation_y;                // Y rotation angle
    float rotation_z;                // Z rotation angle
    float zoom;                      // Zoom level
    bool animate;                    // Animation flag
    int frame_count;                 // Frame counter
} QuantumCubeState;

// Function prototypes
int init_quantum_visualizer(int argc, char** argv);
void update_quantum_visualizer(void);
void add_quantum_node(float x, float y, float z, float size, float r, float g, float b, float energy, int type, const char* label);
void add_quantum_connection(int source, int target, float strength, float r, float g, float b);
void set_rotation(float x, float y, float z);
void set_zoom(float zoom);
void toggle_animation(void);
void shutdown_quantum_visualizer(void);

// Function to update visualization data (if needed externally)
void update_visualization_data(/* parameters for data */);

#endif // QUANTUM_CUBE_VISUALIZER_H
