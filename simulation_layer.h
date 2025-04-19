#ifndef SIMULATION_LAYER_H
#define SIMULATION_LAYER_H

#include <stdint.h>
#include <stdbool.h>

// Simulation entities
typedef struct {
    uint64_t id;
    float position[3];  // x, y, z
    float velocity[3];  // vx, vy, vz
    float properties[10]; // Flexible properties array
    char type[32];
    bool active;
} SimulationEntity;

// Simulation configuration
typedef struct {
    uint32_t max_entities;
    float time_step;
    float space_bounds[3]; // x, y, z bounds
    bool wrap_around;
    uint32_t iterations_per_update;
    void* custom_params;
} SimulationConfig;

// Simulation layer
typedef struct {
    SimulationEntity* entities;
    uint32_t entity_count;
    uint32_t max_entities;
    uint64_t simulation_time;
    uint64_t step_count;
    SimulationConfig config;
    bool is_running;
} SimulationLayer;

// Function prototypes
SimulationLayer* init_simulation(SimulationConfig* config);
int start_simulation(SimulationLayer* layer);
int pause_simulation(SimulationLayer* layer);
int reset_simulation(SimulationLayer* layer);
SimulationEntity* add_entity(SimulationLayer* layer, float position[3], float velocity[3], const char* type);
int remove_entity(SimulationLayer* layer, uint64_t id);
void update_simulation(SimulationLayer* layer);
void free_simulation(SimulationLayer* layer);

#endif // SIMULATION_LAYER_H