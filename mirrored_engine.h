#ifndef MIRRORED_ENGINE_H
#define MIRRORED_ENGINE_H
#include "kaleidoscope_engine.h"
#include "memory_graph.h"
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>

// Structure to represent a mirrored network
typedef struct MirroredNetwork MirroredNetwork;

// Simulation functions
void simulate_compound_interaction(KaleidoscopeEngine* engine, const char* compound_data);
int evaluate_compound(const char* compound_data);

// Mirrored network operations
MirroredNetwork* init_mirrored_network(uint64_t max_nodes);
void destroy_mirrored_network(MirroredNetwork* network);
void generate_computational_suggestion(MirroredNetwork* network, const char* problem_description);
void propose_to_node(MirroredNetwork* network, void* node);

// Error handling
const char* mirrored_engine_get_last_error(void);

#endif // MIRRORED_ENGINE_H

