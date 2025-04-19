#ifndef WEB_CRAWLER_H
#define WEB_CRAWLER_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h> // For potential threading

// Forward declaration only, do not redefine DataIngestionLayer here
#ifndef DATA_INGESTION_LAYER_FWD_DECL
#define DATA_INGESTION_LAYER_FWD_DECL
typedef struct DataIngestionLayer DataIngestionLayer;
#endif

// Crawler Configuration
typedef struct {
    int max_pages_per_domain;
    int crawl_delay_ms;
    int timeout_seconds;
    int max_results;
    int max_crawl_depth;
    bool respect_robots_txt;
    bool follow_redirects;
    char user_agent[128];
} CrawlerConfig;

// Web Crawler Task (if managed separately)
typedef struct {
    char* url;
    int depth;
    // Add other task details
} WebCrawlerTask;

// Web Crawler Structure
typedef struct WebCrawler {
    DataIngestionLayer* ingestion_layer; // Link back to ingestion layer
    CrawlerConfig config;
    // Add state variables: queue, visited URLs, active threads, etc.
    // Example: Queue* task_queue;
    // Example: HashTable* visited_urls;
    pthread_mutex_t lock; // For thread safety
    bool is_running;
} WebCrawler;

// Function Prototypes
WebCrawler* init_web_crawler(DataIngestionLayer* ingestion_layer);
void destroy_web_crawler(WebCrawler* crawler);
void configure_web_crawler(WebCrawler* crawler, const CrawlerConfig* config);

// Starts a crawl based on a search term or starting URL
// Returns true if crawl started successfully, false otherwise.
bool start_crawl(WebCrawler* crawler, const char* start_query);

// Stops the currently running crawl
void stop_crawl(WebCrawler* crawler);

// Function for autonomous crawling based on a topic
// If 'high_priority' is true, it might bypass normal delays/limits.
bool autonomous_data_crawl(WebCrawler* crawler, const char* topic, bool high_priority);

// Gets the last error message
const char* web_crawler_get_last_error(void);

#endif // WEB_CRAWLER_H
