#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Visualization Metrics Structure
typedef struct {
    uint64_t* energy_levels;
    uint64_t* task_completions;
    uint64_t max_cycles;
    uint64_t current_cycle;
} VisualizationTools;

// Function Prototypes
VisualizationTools* init_visualization_tools(uint64_t max_cycles);
void update_metrics(VisualizationTools* tools, uint64_t energy, uint64_t tasks_completed);
void display_metrics(VisualizationTools* tools);
void destroy_visualization_tools(VisualizationTools* tools);

#endif // VISUALIZATION_TOOLS_H

