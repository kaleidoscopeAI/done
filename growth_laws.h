#ifndef GROWTH_LAWS_H
#define GROWTH_LAWS_H

#include <stdbool.h>
#include <stdint.h>

// System parameters structure for growth laws
typedef struct {
    float mutation_rate;
    float energy_transfer_ratio;
    float stress_spread_factor;
    float novelty_reward;
    float complexity_cost;
} SystemParameters;

// GrowthLaws structure
typedef struct {
    // Constants for energy dynamics
    float ENERGY_THRESHOLD_REPRODUCE;
    float MIN_STABILITY_REPRODUCE;
    float ENERGY_DECAY_BASE;
    float STRESS_RECOVERY_RATE;
    
    // Dynamic parameters (can evolve over time)
    SystemParameters system_parameters;
} GrowthLaws;

// Node state structure (needed for growth laws operations)
typedef struct {
    float energy;
    float stability;
    float stress_level;
    bool ready_to_replicate;
    char id[64];  // Node identifier
} NodeState;

// Function prototypes
GrowthLaws* growth_laws_init();
void growth_laws_apply(GrowthLaws* laws, NodeState** nodes, int node_count, float external_stress);
void growth_laws_apply_to_node(GrowthLaws* laws, NodeState* node, float external_stress);
void growth_laws_destroy(GrowthLaws* laws);

#endif /* GROWTH_LAWS_H */