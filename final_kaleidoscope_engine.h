#ifndef FINAL_KALEIDOSCOPE_ENGINE_H
#define FINAL_KALEIDOSCOPE_ENGINE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "memory_graph.h"

// Final Kaleidoscope Engine Structure
typedef struct {
    MemoryGraph* memory_graph;
    uint64_t master_insight_count;
} FinalEngine;

// Function Prototypes
FinalEngine* init_final_engine(MemoryGraph* memory_graph);
void generate_master_insight(FinalEngine* engine, const char* meta_insight);
void destroy_final_engine(FinalEngine* engine);

#endif // FINAL_KALEIDOSCOPE_ENGINE_H