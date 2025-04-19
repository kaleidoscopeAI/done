#ifndef NODE_DNA_H
#define NODE_DNA_H

#include <stdint.h>
#include <stdbool.h>

#define DNA_SEQUENCE_MAX_LEN 1024

// Structure representing Node DNA
typedef struct {
    uint64_t node_id; // ID of the node this DNA belongs to
    char sequence[DNA_SEQUENCE_MAX_LEN]; // The DNA sequence string
    uint32_t length;
    uint64_t generation; // Generation number of this DNA
    // Add fields for mutation rate, fitness score, parent IDs, etc.
    double mutation_rate;
    double fitness_score;
} NodeDNA;

// Function Prototypes
NodeDNA* create_node_dna(uint64_t node_id, const char* initial_sequence);
void destroy_node_dna(NodeDNA* dna);

// Mutates the DNA sequence based on its mutation rate
void mutate_dna(NodeDNA* dna);

// Combines DNA from two parents to create offspring DNA
NodeDNA* crossover_dna(const NodeDNA* parent1, const NodeDNA* parent2, uint64_t child_node_id);

// Calculates the fitness of the DNA based on some criteria (placeholder)
double calculate_dna_fitness(const NodeDNA* dna);

// Gets the last error message related to DNA operations
const char* node_dna_get_last_error(void);

#endif // NODE_DNA_H
