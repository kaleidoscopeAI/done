#ifndef AUTONOMOUS_LEARNING_H
#define AUTONOMOUS_LEARNING_H

#include <stdint.h>
#include <stdbool.h>

// Forward declarations only, do not redefine these structs here
#ifndef AUTONOMOUS_LEARNING_FWD_DECLS
#define AUTONOMOUS_LEARNING_FWD_DECLS
typedef struct DataIngestionLayer DataIngestionLayer;
typedef struct WebCrawler WebCrawler;
typedef struct MemoryGraph MemoryGraph;
typedef struct FeedbackCollector FeedbackCollector;
#endif

// Structure for knowledge gaps
typedef struct {
    uint64_t id;
    char* topic;
    double priority;
    uint64_t identified_timestamp;
} KnowledgeGap;

// Autonomous Learning System structure
typedef struct {
    DataIngestionLayer* ingestion_layer;
    WebCrawler* crawler;
    MemoryGraph* memory_graph;
    FeedbackCollector* feedback_system;
    KnowledgeGap** knowledge_gaps;
    uint32_t gap_count;
    uint32_t gap_capacity;
    // Add state, learning models, parameters, etc.
    uint64_t last_process_time;
} AutonomousLearning;

// Function Prototypes
AutonomousLearning* init_autonomous_learning(
    DataIngestionLayer* ingestion,
    WebCrawler* crawler,
    MemoryGraph* graph,
    FeedbackCollector* feedback
);
void destroy_autonomous_learning(AutonomousLearning* system);

// Main processing function for the learning system
void autonomous_learning_process(AutonomousLearning* system);

// Identifies knowledge gaps based on current graph state and feedback
void identify_knowledge_gaps(AutonomousLearning* system);

// Attempts to fill identified knowledge gaps (e.g., by triggering crawls)
void fill_knowledge_gaps(AutonomousLearning* system);

// Gets the current count of identified knowledge gaps
uint32_t get_knowledge_gap_count(AutonomousLearning* system);

// Gets the last error message
const char* autonomous_learning_get_last_error(void);

#endif // AUTONOMOUS_LEARNING_H
