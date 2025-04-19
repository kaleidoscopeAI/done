#ifndef MEMBRANE_H
#define MEMBRANE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Define the membrane node structure
typedef struct MembraneNode {
    uint64_t id;
    uint64_t energy;
    struct MembraneNode* next;
} MembraneNode;

// Define the membrane structure
typedef struct Membrane {
    MembraneNode** nodes;
    uint64_t node_count;
    uint64_t max_nodes;
    uint64_t total_energy;
} Membrane;

// Function declarations - only declarations, not implementations
Membrane* create_membrane(uint64_t initial_energy, uint64_t max_nodes);
int add_node_to_membrane(Membrane* membrane, MembraneNode* node);
void distribute_energy(Membrane* membrane);
MembraneNode* isolate_node(Membrane* membrane, uint64_t node_id);
void replicate_node_in_membrane(Membrane* membrane, uint64_t node_id);
void replicate_nodes(Membrane* membrane);
void destroy_membrane(Membrane* membrane);

#endif // MEMBRANE_H

