#ifndef PERSPECTIVE_MANAGER_H
#define PERSPECTIVE_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include "memory_graph.h"
#include "common_types.h"

// Perspective structure
typedef struct {
    uint64_t id;
    char* name;
    char* description;
    void* data;         // Flexible data pointer for perspective-specific information
    uint32_t ref_count; // Reference counting for memory management
    float* weights;     // Weights array for this perspective
    uint32_t weight_count; // Number of weights
    bool active;        // Whether this perspective is active
} Perspective;

// Perspective manager to handle multiple perspectives
typedef struct {
    Perspective** perspectives;    // Array of perspective pointers
    uint32_t perspective_count;    // Current number of perspectives
    uint32_t max_perspectives;     // Maximum capacity
    uint64_t next_perspective_id;  // Counter for generating unique IDs
} PerspectiveManager;

// Function prototypes
PerspectiveManager* init_perspective_manager(uint32_t max_perspectives);
uint64_t create_perspective(PerspectiveManager* manager, const char* name, const char* description);
Perspective* get_perspective(PerspectiveManager* manager, uint64_t id);
int update_perspective_info(PerspectiveManager* manager, uint64_t id, const char* name, const char* description);
int delete_perspective(PerspectiveManager* manager, uint64_t id);
bool activate_perspective(PerspectiveManager* manager, uint64_t id);
bool deactivate_perspective(PerspectiveManager* manager, uint64_t id);
bool set_perspective_weights(PerspectiveManager* manager, uint64_t id, const float* weights, uint32_t count);
bool link_perspective_to_memory_graph(PerspectiveManager* manager, uint64_t id, MemoryGraph* graph);
void free_perspective_manager(PerspectiveManager* manager);

bool initialize_perspective(Perspective* perspective, float* initial_weights, int weight_count);
bool update_perspective_state(Perspective* perspective, MemoryGraph* memory_graph);

#endif // PERSPECTIVE_MANAGER_H