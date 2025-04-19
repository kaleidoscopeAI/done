#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "feedback_system.h"
#include "bridge_adapter.h"
#include "data_ingestion.h"

// Global instance
FeedbackCollector* g_feedback_system = NULL;

// Error handling
static void set_feedback_error(FeedbackCollector* collector, const char* format, ...) {
    if (!collector) return;
    va_list args;
    va_start(args, format);
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    snprintf(collector->last_error, sizeof(collector->last_error), "[%s] %s", timestamp, format);
    vsnprintf(collector->last_error, sizeof(collector->last_error), collector->last_error, args);
    va_end(args);
    fprintf(stderr, "[Feedback System Error] %s\n", collector->last_error);
}

const char* feedback_system_get_last_error(FeedbackCollector* collector) {
    return collector ? collector->last_error : "Feedback system not initialized";
}

FeedbackCollector* init_feedback_system(uint64_t max_feedback, bool logging_enabled, const char* log_path) {
    if (max_feedback == 0) {
        fprintf(stderr, "[Feedback System Error] Invalid max_feedback value\n");
        return NULL;
    }

    FeedbackCollector* collector = (FeedbackCollector*)calloc(1, sizeof(FeedbackCollector));
    if (!collector) {
        fprintf(stderr, "[Feedback System Error] Failed to allocate memory for feedback collector\n");
        return NULL;
    }
    
    collector->recent_feedback = NULL;
    collector->count = 0;
    collector->applied_count = 0;
    collector->system_health_score = 0.8;
    collector->logging_enabled = logging_enabled;
    collector->log_path = log_path ? strdup(log_path) : NULL;
    if (log_path && !collector->log_path) {
        set_feedback_error(collector, "Failed to allocate memory for log path");
        free(collector);
        return NULL;
    }
    time_t now = time(NULL);
    strftime(collector->last_error, sizeof(collector->last_error), "[%Y-%m-%d %H:%M:%S] No error", localtime(&now));
    
    collector->learning_system = init_learning_feedback_system(max_feedback, 0.01);
    if (!collector->learning_system) {
        set_feedback_error(collector, "Failed to initialize learning feedback system");
        free(collector->log_path);
        free(collector);
        return NULL;
    }
    
    g_feedback_system = collector;
    printf("Feedback system initialized with capacity for %lu feedback entries\n", max_feedback);
    return collector;
}

FeedbackStatus add_feedback(FeedbackCollector* collector, FeedbackSourceType source_type, uint64_t source_id,
                           FeedbackTargetType target_type, uint64_t target_id, double score,
                           FeedbackImpactLevel impact, const char* description) {
    if (!collector || !description) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    if (score < -1.0 || score > 1.0) {
        set_feedback_error(collector, "Feedback score %f out of range, clamping", score);
        score = (score < -1.0) ? -1.0 : (score > 1.0 ? 1.0 : score);
        return FEEDBACK_INVALID_SCORE;
    }
    
    FeedbackEntry* entry = (FeedbackEntry*)calloc(1, sizeof(FeedbackEntry));
    if (!entry) {
        set_feedback_error(collector, "Failed to allocate memory for feedback entry");
        return FEEDBACK_MEMORY_ERROR;
    }
    
    entry->id = collector->count + 1;
    entry->source_type = source_type;
    entry->source_id = source_id;
    entry->target_type = target_type;
    entry->target_id = target_id;
    entry->score = score;
    entry->impact = impact;
    entry->timestamp = time(NULL);
    entry->applied_timestamp = 0;
    strncpy(entry->description, description, sizeof(entry->description) - 1);
    entry->description[sizeof(entry->description) - 1] = '\0';
    
    entry->next = collector->recent_feedback;
    collector->recent_feedback = entry;
    collector->count++;
    
    if (record_feedback(collector->learning_system, score) != 0) {
        set_feedback_error(collector, "Failed to record feedback in learning system");
        free(entry);
        collector->recent_feedback = entry->next;
        collector->count--;
        return FEEDBACK_MEMORY_ERROR;
    }
    
    double impact_weight = (double)impact / 10.0;
    collector->system_health_score = collector->system_health_score * 0.9 + (score * impact_weight) * 0.1;
    if (collector->system_health_score < 0.0) collector->system_health_score = 0.0;
    if (collector->system_health_score > 1.0) collector->system_health_score = 1.0;
    
    if (collector->logging_enabled && collector->log_path) {
        FILE* log_file = fopen(collector->log_path, "a");
        if (!log_file) {
            set_feedback_error(collector, "Failed to open log file: %s", collector->log_path);
            return FEEDBACK_LOGGING_ERROR;
        }
        fprintf(log_file, "[%lu] Source: %d, Target: %d, Score: %.2f, Impact: %d, Description: %s\n",
                entry->timestamp, source_type, target_type, score, impact, description);
        fclose(log_file);
    }
    
    printf("Added feedback (%lu): %s (score: %.2f, impact: %d)\n", entry->id, description, score, impact);
    return FEEDBACK_OK;
}

FeedbackStatus process_all_feedback(FeedbackCollector* collector, MemoryGraph* graph) {
    if (!collector || !graph) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    
    printf("Processing all pending feedback...\n");
    
    double adjusted_learning_rate = calculate_adjusted_learning_rate(collector->learning_system);
    uint64_t applied_count = 0;
    
    FeedbackEntry* current = collector->recent_feedback;
    while (current) {
        if (current->applied_timestamp == 0) {
            FeedbackStatus status = FEEDBACK_OK;
            switch (current->target_type) {
                case FEEDBACK_TARGET_SUPERNODE: {
                    ConsciousSuperNode* supernode = (ConsciousSuperNode*)(uintptr_t)current->target_id;
                    if (supernode) {
                        status = apply_feedback_to_supernode(collector, current, supernode);
                        if (status == FEEDBACK_OK) applied_count++;
                    } else {
                        set_feedback_error(collector, "Invalid supernode for feedback %lu", current->id);
                        status = FEEDBACK_NULL_PTR;
                    }
                    break;
                }
                case FEEDBACK_TARGET_NODE: {
                    MemoryNode* node = get_node(graph, current->target_id);
                    if (node) {
                        status = apply_feedback_to_node(collector, current, node);
                        if (status == FEEDBACK_OK) applied_count++;
                    } else {
                        set_feedback_error(collector, "Node %lu not found for feedback %lu", current->target_id, current->id);
                        status = FEEDBACK_NULL_PTR;
                    }
                    break;
                }
                case FEEDBACK_TARGET_CONNECTION: {
                    uint64_t source_id = 0, target_id = 0;
                    if (sscanf(current->description, "connection:%lu:%lu", &source_id, &target_id) == 2) {
                        status = apply_feedback_to_connection(collector, current, graph, source_id, target_id);
                        if (status == FEEDBACK_OK) applied_count++;
                    } else {
                        set_feedback_error(collector, "Invalid connection format for feedback %lu", current->id);
                        status = FEEDBACK_NULL_PTR;
                    }
                    break;
                }
                case FEEDBACK_TARGET_SYSTEM: {
                    status = apply_system_feedback(collector, current);
                    if (status == FEEDBACK_OK) applied_count++;
                    break;
                }
                case FEEDBACK_TARGET_ALGORITHM: {
                    status = integrate_learning_feedback(collector, current->score * 0.1);
                    if (status == FEEDBACK_OK) applied_count++;
                    break;
                }
            }
            if (status == FEEDBACK_OK) {
                current->applied_timestamp = time(NULL);
                
                if (collector->logging_enabled && collector->log_path) {
                    FILE* log_file = fopen(collector->log_path, "a");
                    if (!log_file) {
                        set_feedback_error(collector, "Failed to open log file: %s", collector->log_path);
                    } else {
                        fprintf(log_file, "[%lu] Applied feedback %lu\n", current->applied_timestamp, current->id);
                        fclose(log_file);
                    }
                }
            }
        }
        current = current->next;
    }
    
    collector->applied_count += applied_count;
    printf("Applied %lu feedback entries (total: %lu)\n", applied_count, collector->applied_count);
    return FEEDBACK_OK;
}

FeedbackStatus apply_feedback_to_supernode(FeedbackCollector* collector, FeedbackEntry* feedback, ConsciousSuperNode* supernode) {
    if (!collector || !feedback || !supernode) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    
    printf("Applying feedback to SuperNode %lu: %s\n", feedback->target_id, feedback->description);
    
    double adjustment = feedback->score * (double)feedback->impact / 10.0;
    
    if (strstr(feedback->description, "learning")) {
        supernode->core_state.evolution_rate *= (1.0 + adjustment);
        printf("  Adjusted evolution rate to %.2f\n", supernode->core_state.evolution_rate);
    } else if (strstr(feedback->description, "energy")) {
        supernode->core_state.energy *= (1.0 + adjustment);
        printf("  Adjusted energy to %.2f\n", supernode->core_state.energy);
    } else if (strstr(feedback->description, "awareness")) {
        supernode->core_state.awareness_level *= (1.0 + adjustment);
        printf("  Adjusted awareness level to %.2f\n", supernode->core_state.awareness_level);
    } else if (strstr(feedback->description, "adaptation")) {
        supernode->core_state.adaptation_rate *= (1.0 + adjustment);
        printf("  Adjusted adaptation rate to %.2f\n", supernode->core_state.adaptation_rate);
    }

    if (supernode->memory_graph) {
        char json_data[512];
        snprintf(json_data, sizeof(json_data), "{\"feedback_id\":%lu,\"adjustment\":%.2f}", feedback->id, adjustment);
        if (bridge_update_memory_graph(supernode->node_id, json_data) != 0) {
            set_feedback_error(collector, "Failed to update memory graph for supernode %lu", supernode->node_id);
        }
    }
    
    return FEEDBACK_OK;
}

FeedbackStatus apply_feedback_to_node(FeedbackCollector* collector, FeedbackEntry* feedback, MemoryNode* node) {
    if (!collector || !feedback || !node) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    
    printf("Applying feedback to Node %lu: %s\n", node->id, feedback->description);
    
    double adjustment = feedback->score * (double)feedback->impact / 10.0;
    float new_activation = node->activation * (1.0f + (float)adjustment);
    
    if (new_activation < 0.0f) new_activation = 0.0f;
    if (new_activation > 1.0f) new_activation = 1.0f;
    
    node->activation = new_activation;
    printf("  Adjusted node activation to %.2f\n", node->activation);

    char json_data[512];
    snprintf(json_data, sizeof(json_data), "{\"feedback_id\":%lu,\"activation\":%.2f}", feedback->id, new_activation);
    if (bridge_update_memory_graph(node->id, json_data) != 0) {
        set_feedback_error(collector, "Failed to update memory graph for node %lu", node->id);
    }
    
    return FEEDBACK_OK;
}

FeedbackStatus apply_feedback_to_connection(FeedbackCollector* collector, FeedbackEntry* feedback,
                                          MemoryGraph* graph, uint64_t source_id, uint64_t target_id) {
    if (!collector || !feedback || !graph) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    
    printf("Applying feedback to connection between %lu and %lu: %s\n", source_id, target_id, feedback->description);
    
    MemoryNode* source = get_node(graph, source_id);
    MemoryNode* target = get_node(graph, target_id);
    
    if (!source || !target) {
        set_feedback_error(collector, "Source or target node not found");
        return FEEDBACK_NULL_PTR;
    }
    
    float activation_boost = (float)(feedback->score * (double)feedback->impact / 10.0);
    float propagated_activation = source->activation * (0.5f + activation_boost);
    
    target->activation += propagated_activation;
    if (target->activation < 0.0f) target->activation = 0.0f;
    if (target->activation > 1.0f) target->activation = 1.0f;
    
    printf("  Propagated %.2f activation from node %lu to node %lu\n", propagated_activation, source_id, target_id);

    char json_data[512];
    snprintf(json_data, sizeof(json_data), "{\"feedback_id\":%lu,\"source\":%lu,\"target\":%lu,\"activation\":%.2f}",
             feedback->id, source_id, target_id, target->activation);
    if (bridge_update_memory_graph(target_id, json_data) != 0) {
        set_feedback_error(collector, "Failed to update memory graph for connection %lu->%lu", source_id, target_id);
    }
    
    return FEEDBACK_OK;
}

FeedbackStatus apply_system_feedback(FeedbackCollector* collector, FeedbackEntry* feedback) {
    if (!collector || !feedback) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    
    printf("Applying system-wide feedback: %s\n", feedback->description);
    
    double adjustment = feedback->score * (double)feedback->impact / 20.0;
    FeedbackStatus status = integrate_learning_feedback(collector, adjustment);
    if (status != FEEDBACK_OK) {
        set_feedback_error(collector, "Failed to integrate system feedback");
    }
    
    return status;
}

double get_system_health_score(FeedbackCollector* collector) {
    if (!collector) {
        set_feedback_error(collector, "Collector not initialized");
        return 0.0;
    }
    return collector->system_health_score;
}

uint64_t get_feedback_count(FeedbackCollector* collector) {
    if (!collector) {
        set_feedback_error(collector, "Collector not initialized");
        return 0;
    }
    return collector->count;
}

FeedbackEntry* get_recent_feedback(FeedbackCollector* collector, uint64_t count) {
    if (!collector || count == 0) {
        set_feedback_error(collector, "Invalid arguments");
        return NULL;
    }
    
    FeedbackEntry* current = collector->recent_feedback;
    uint64_t filled = 0;
    
    while (current && filled < count) {
        filled++;
        if (filled == count) {
            FeedbackEntry* next = current->next;
            current->next = NULL;
            FeedbackEntry* result = collector->recent_feedback;
            collector->recent_feedback = next;
            return result;
        }
        current = current->next;
    }
    
    return collector->recent_feedback;
}

FeedbackStatus integrate_learning_feedback(FeedbackCollector* collector, double learning_adjustment) {
    if (!collector || !collector->learning_system) {
        set_feedback_error(collector, "Invalid arguments");
        return FEEDBACK_NULL_PTR;
    }
    
    double current_rate = collector->learning_system->learning_rate;
    double new_rate = current_rate * (1.0 + learning_adjustment);
    if (new_rate < 0.001) new_rate = 0.001;
    if (new_rate > 0.1) new_rate = 0.1;
    
    collector->learning_system->learning_rate = new_rate;
    printf("Adjusted system learning rate: %.4f -> %.4f\n", current_rate, new_rate);
    return FEEDBACK_OK;
}

char* get_feedback_summary(FeedbackCollector* collector) {
    if (!collector) {
        set_feedback_error(collector, "Collector not initialized");
        return NULL;
    }
    
    char* summary = (char*)calloc(4096, sizeof(char));
    if (!summary) {
        set_feedback_error(collector, "Failed to allocate memory for summary");
        return NULL;
    }
    
    size_t offset = snprintf(summary, 4096, "Feedback Summary:\nTotal Feedback: %lu\nApplied Feedback: %lu\nSystem Health Score: %.2f\n",
                            collector->count, collector->applied_count, collector->system_health_score);
    
    if (collector->count == 0) {
        offset += snprintf(summary + offset, 4096 - offset, "No feedback entries available\n");
    } else {
        FeedbackEntry* current = collector->recent_feedback;
        int count = 0;
        while (current && count < 5) {
            offset += snprintf(summary + offset, 4096 - offset, "- [%lu] %s (Score: %.2f, Impact: %d)\n",
                              current->id, current->description, current->score, current->impact);
            current = current->next;
            count++;
        }
        if (collector->count > 5)