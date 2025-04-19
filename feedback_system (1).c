#ifndef FEEDBACK_SYSTEM_H
#define FEEDBACK_SYSTEM_H

#include <stdint.h>
#include <stdbool.h>
#include "memory_graph.h"
#include "conscious_supernode.h"
#include "learning_feedback.h"

// Error codes
typedef enum {
    FEEDBACK_OK = 0,
    FEEDBACK_NULL_PTR = 1,
    FEEDBACK_MEMORY_ERROR = 2,
    FEEDBACK_INVALID_SCORE = 3,
    FEEDBACK_LOGGING_ERROR = 4,
    FEEDBACK_NOT_INITIALIZED = 5
} FeedbackStatus;

// Feedback source types
typedef enum {
    FEEDBACK_SOURCE_INTERNAL,
    FEEDBACK_SOURCE_EXTERNAL,
    FEEDBACK_SOURCE_SIMULATION,
    FEEDBACK_SOURCE_LLM,
    FEEDBACK_SOURCE_NODE
} FeedbackSourceType;

// Feedback target types
typedef enum {
    FEEDBACK_TARGET_SYSTEM,
    FEEDBACK_TARGET_SUPERNODE,
    FEEDBACK_TARGET_NODE,
    FEEDBACK_TARGET_CONNECTION,
    FEEDBACK_TARGET_ALGORITHM
} FeedbackTargetType;

// Feedback impact levels
typedef enum {
    FEEDBACK_IMPACT_NEGLIGIBLE = 1,
    FEEDBACK_IMPACT_LOW = 2,
    FEEDBACK_IMPACT_MEDIUM = 5,
    FEEDBACK_IMPACT_HIGH = 8,
    FEEDBACK_IMPACT_CRITICAL = 10
} FeedbackImpactLevel;

// Feedback data structure
typedef struct FeedbackEntry {
    uint64_t id;
    FeedbackSourceType source_type;
    uint64_t source_id;
    FeedbackTargetType target_type;
    uint64_t target_id;
    double score;
    FeedbackImpactLevel impact;
    char description[256];
    uint64_t timestamp;
    uint64_t applied_timestamp;
    struct FeedbackEntry* next;
} FeedbackEntry;

// Feedback collector structure
typedef struct {
    FeedbackEntry* recent_feedback;
    uint64_t count;
    uint64_t applied_count;
    double system_health_score;
    LearningFeedbackSystem* learning_system;
    char last_error[256];
    bool logging_enabled;
    char* log_path;
} FeedbackCollector;

// Function prototypes
FeedbackCollector* init_feedback_system(uint64_t max_feedback, bool logging_enabled, const char* log_path);
FeedbackStatus add_feedback(FeedbackCollector* collector, FeedbackSourceType source_type, uint64_t source_id,
                           FeedbackTargetType target_type, uint64_t target_id, double score,
                           FeedbackImpactLevel impact, const char* description);
FeedbackStatus process_all_feedback(FeedbackCollector* collector, MemoryGraph* graph);
FeedbackStatus apply_feedback_to_supernode(FeedbackCollector* collector, FeedbackEntry* feedback, ConsciousSuperNode* supernode);
FeedbackStatus apply_feedback_to_node(FeedbackCollector* collector, FeedbackEntry* feedback, MemoryNode* node);
FeedbackStatus apply_feedback_to_connection(FeedbackCollector* collector, FeedbackEntry* feedback,
                                          MemoryGraph* graph, uint64_t source_id, uint64_t target_id);
FeedbackStatus apply_system_feedback(FeedbackCollector* collector, FeedbackEntry* feedback);
double get_system_health_score(FeedbackCollector* collector);
uint64_t get_feedback_count(FeedbackCollector* collector);
FeedbackEntry* get_recent_feedback(FeedbackCollector* collector, uint64_t count);
FeedbackStatus integrate_learning_feedback(FeedbackCollector* collector, double learning_adjustment);
char* get_feedback_summary(FeedbackCollector* collector);
const char* feedback_system_get_last_error(FeedbackCollector* collector);
void destroy_feedback_system(FeedbackCollector* collector);

// Global instance
extern FeedbackCollector* g_feedback_system;

#endif // FEEDBACK_SYSTEM_H