#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "environment_simulation.h"
#include "bridge_adapter.h"

// Error handling
static void set_simulation_error(EnvironmentSimulation* simulation, const char* format, ...) {
    if (!simulation) return;
    va_list args;
    va_start(args, format);
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    snprintf(simulation->last_error, sizeof(simulation->last_error), "[%s] %s", timestamp, format);
    vsnprintf(simulation->last_error, sizeof(simulation->last_error), simulation->last_error, args);
    va_end(args);
    fprintf(stderr, "[Environment Simulation Error] %s\n", simulation->last_error);
}

const char* environment_simulation_get_last_error(EnvironmentSimulation* simulation) {
    return simulation ? simulation->last_error : "Simulation not initialized";
}

EnvironmentSimulation* init_environment_simulation(uint64_t max_states, MemoryGraph* memory_graph,
                                                  ConsciousSuperNode* supernode, DataIngestionLayer* ingestion_layer) {
    if (max_states == 0) {
        fprintf(stderr, "[Environment Simulation Error] Invalid max_states value\n");
        return NULL;
    }

    EnvironmentSimulation* simulation = (EnvironmentSimulation*)calloc(1, sizeof(EnvironmentSimulation));
    if (!simulation) {
        fprintf(stderr, "[Environment Simulation Error] Failed to allocate memory for Environment Simulation\n");
        return NULL;
    }

    simulation->states = (EnvironmentState*)calloc(max_states, sizeof(EnvironmentState));
    if (!simulation->states) {
        set_simulation_error(simulation, "Failed to allocate memory for states");
        free(simulation);
        return NULL;
    }

    simulation->state_count = 0;
    simulation->max_states = max_states;
    simulation->memory_graph = memory_graph;
    simulation->supernode = supernode;
    simulation->ingestion_layer = ingestion_layer;
    simulation->is_running = true;
    time_t now = time(NULL);
    strftime(simulation->last_error, sizeof(simulation->last_error), "[%Y-%m-%d %H:%M:%S] No error", localtime(&now));

    printf("Environment Simulation initialized with capacity for %lu states\n", max_states);
    return simulation;
}

EnvironmentSimulationStatus update_environment_state(EnvironmentSimulation* simulation, uint64_t data, uint64_t conditions, const char* description) {
    if (!simulation || !description) {
        set_simulation_error(simulation, "Invalid arguments");
        return ENV_SIM_NULL_PTR;
    }
    if (simulation->state_count >= simulation->max_states) {
        set_simulation_error(simulation, "State capacity reached");
        return ENV_SIM_CAPACITY_EXCEEDED;
    }

    EnvironmentState* state = &simulation->states[simulation->state_count++];
    state->external_data = data;
    state->external_conditions = conditions;
    state->interaction_count = 0;
    state->stability_score = 1.0;
    strncpy(state->description, description, sizeof(state->description) - 1);
    state->description[sizeof(state->description) - 1] = '\0';

    if (simulation->memory_graph) {
        char json_data[512];
        snprintf(json_data, sizeof(json_data), "{\"data\":%lu,\"conditions\":%lu,\"description\":\"%s\"}",
                 data, conditions, description);
        MemoryNode* node = create_memory_node(json_data, 1.0);
        if (!node || add_memory_node(simulation->memory_graph, node) != 0) {
            if (node) destroy_memory_node(node);
            set_simulation_error(simulation, "Failed to add state to memory graph");
            simulation->state_count--;
            return ENV_SIM_MEMORY_ERROR;
        }
    }

    if (simulation->supernode) {
        absorb_knowledge(simulation->supernode, description);
    }

    if (simulation->ingestion_layer) {
        DataIngestionStatus status = ingest_text(simulation->ingestion_layer, description);
        if (status != DATA_INGESTION_OK) {
            set_simulation_error(simulation, "Failed to ingest state description");
            simulation->state_count--;
            return ENV_SIM_MEMORY_ERROR;
        }
    }

    printf("Environment State Updated: Data=%lu, Conditions=%lu, Description=%s\n", data, conditions, description);
    return ENV_SIM_OK;
}

EnvironmentSimulationStatus simulate_interaction(EnvironmentSimulation* simulation, uint64_t node_id) {
    if (!simulation) {
        set_simulation_error(simulation, "Simulation not initialized");
        return ENV_SIM_NULL_PTR;
    }
    if (simulation->state_count == 0) {
        set_simulation_error(simulation, "No states available");
        return ENV_SIM_NO_STATES;
    }
    if (!simulation->is_running) {
        set_simulation_error(simulation, "Simulation not running");
        return ENV_SIM_NOT_RUNNING;
    }

    // Select the most recent state
    EnvironmentState* current_state = &simulation->states[simulation->state_count - 1];
    current_state->interaction_count++;
    current_state->stability_score *= 0.99; // Gradual decay

    if (simulation->memory_graph) {
        char json_data[512];
        snprintf(json_data, sizeof(json_data), "{\"node_id\":%lu,\"interaction\":%lu,\"stability\":%.2f}",
                 node_id, current_state->interaction_count, current_state->stability_score);
        if (bridge_update_memory_graph(node_id, json_data) != 0) {
            set_simulation_error(simulation, "Failed to update memory graph for interaction");
        }
    }

    if (current_state->stability_score < 0.5 && simulation->ingestion_layer) {
        DataIngestionStatus status = check_data_need(simulation->ingestion_layer, current_state->description);
        if (status != DATA_INGESTION_OK) {
            set_simulation_error(simulation, "Failed to check data need for unstable state");
        }
    }

    printf("Node %lu interacting with environment: Data=%lu, Conditions=%lu, Interactions=%lu, Stability=%.2f\n",
           node_id, current_state->external_data, current_state->external_conditions,
           current_state->interaction_count, current_state->stability_score);
    return ENV_SIM_OK;
}

void display_environment(EnvironmentSimulation* simulation) {
    if (!simulation) {
        printf("Error: Simulation not initialized\n");
        return;
    }

    printf("\n--- Environment States ---\n");
    if (simulation->state_count == 0) {
        printf("No states available\n");
    } else {
        for (uint64_t i = 0; i < simulation->state_count; i++) {
            EnvironmentState* state = &simulation->states[i];
            printf("State %lu: Data=%lu, Conditions=%lu, Interactions=%lu, Stability=%.2f, Description=%s\n",
                   i + 1, state->external_data, state->external_conditions, state->interaction_count,
                   state->stability_score, state->description);
        }
    }
    printf("---------------------------\n");
}

void destroy_environment_simulation(EnvironmentSimulation* simulation) {
    if (!simulation) return;

    free(simulation->states);
    free(simulation);
    printf("Environment Simulation destroyed\n");
}