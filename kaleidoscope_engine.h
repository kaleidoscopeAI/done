#ifndef KALEIDOSCOPE_ENGINE_H
#define KALEIDOSCOPE_ENGINE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "pattern_recognition.h"
#include "optimization.h"

// Kaleidoscope Engine Structure
typedef struct {
    PatternRecognizer* recognizer;
    Optimizer* optimizer;
} KaleidoscopeEngine;

// Function Prototypes
KaleidoscopeEngine* init_kaleidoscope_engine(void);
void process_task(KaleidoscopeEngine* engine, const char* task_data);
void destroy_kaleidoscope_engine(KaleidoscopeEngine* engine);
void shutdown_kaleidoscope_engine(void);

#endif // KALEIDOSCOPE_ENGINE_H
