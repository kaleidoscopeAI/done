#ifndef ENVIRONMENT_SIMULATION_H
#define ENVIRONMENT_SIMULATION_H

#include <stdint.h>
#include <stdbool.h>
#include "memory_graph.h"
#include "conscious_supernode.h"
#include "data_ingestion.h"

// Error codes
typedef enum {
    ENV_SIM_OK = 0,
    ENV_SIM_NULL_PTR = 1,
    ENV_SIM_MEMORY_ERROR = 2,
    ENV_SIM_CAPACITY_EXCEEDED = 3,
    ENV_SIM_NOT_RUNNING = 4,
    ENV_SIM_NO_STATES = 5
} EnvironmentSimulationStatus;

// Environment State Structure
typedef struct {
    uint64_t external_data;
    uint64_t external_conditions;
    uint64_t interaction_count;
    double stability_score;
    char description[256];
} EnvironmentState;

// Environment Simulation Structure
typedef struct {
    EnvironmentState* states;
    uint64_t state_count;
    uint64_t max_states;
    MemoryGraph* memory_graph;
    ConsciousSuperNode* supernode;
    DataIngestionLayer* ingestion_layer;
    bool is_running;
    char last_error[256];
} EnvironmentSimulation;

// Function Prototypes
EnvironmentSimulation* init_environment_simulation(uint64_t max_states, MemoryGraph* memory_graph,
                                                  ConsciousSuperNode* supernode, DataIngestionLayer* ingestion_layer);
EnvironmentSimulationStatus update_environment_state(EnvironmentSimulation* simulation, uint64_t data, uint64_t conditions, const char* description);
EnvironmentSimulationStatus simulate_interaction(EnvironmentSimulation* simulation, uint64_t node_id);
void display_environment(EnvironmentSimulation* simulation);
const char* environment_simulation_get_last_error(EnvironmentSimulation* simulation);
void destroy_environment_simulation(EnvironmentSimulation* simulation);

#endif // ENVIRONMENT_SIMULATION_H