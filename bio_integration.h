#ifndef BIO_INTEGRATION_H
#define BIO_INTEGRATION_H

#include <stdint.h>
#include <stdbool.h>

// Structure for biological data representation
typedef struct {
    char* sequence_id;
    char* sequence_data; // e.g., DNA, RNA, protein sequence
    char* metadata; // e.g., source organism, function
} BioData;

// Structure for the bio integration module
typedef struct {
    // Configuration, models, databases related to biological data
    char* database_path;
    double similarity_threshold;
} BioIntegrationModule;

// Function Prototypes
BioIntegrationModule* init_bio_integration(const char* db_path);
void destroy_bio_integration(BioIntegrationModule* module);

// Processes biological data, potentially adding it to the memory graph or triggering analysis
bool process_bio_data(BioIntegrationModule* module, const BioData* data);

// Searches for similar biological sequences or patterns
BioData** find_similar_bio_data(BioIntegrationModule* module, const char* query_sequence, uint32_t* result_count);

// Frees the results from find_similar_bio_data
void free_bio_data_results(BioData** results, uint32_t count);

// Gets the last error message
const char* bio_integration_get_last_error(void);


#endif // BIO_INTEGRATION_H
