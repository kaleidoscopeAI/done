#ifndef WEBSOCKET_SERVER_H
#define WEBSOCKET_SERVER_H

#include <stdint.h>
#include <stdbool.h>

// WebSocket connection
typedef struct {
    void* wsi;
    uint64_t id;
    char* ip_address;
    uint64_t connect_time;
    bool authenticated;
} WebSocketConnection;

// WebSocket message
typedef struct {
    uint64_t id;
    uint64_t connection_id;
    char* payload;
    uint32_t length;
    uint64_t timestamp;
    bool is_binary;
} WebSocketMessage;

// Function prototypes
int init_websocket_server(void);
void service_websocket(void);
int send_message(uint64_t connection_id, const char* payload, uint32_t length, bool is_binary);
int broadcast_message(const char* payload, uint32_t length, bool is_binary);
int authenticate_connection(uint64_t connection_id, const char* auth_token);
void destroy_websocket_server(void);

#endif // WEBSOCKET_SERVER_H
