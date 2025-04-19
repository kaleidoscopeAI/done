#ifndef WEBSOCKET_SERVER_STUB_H
#define WEBSOCKET_SERVER_STUB_H

#include <stdint.h>
#include <stdbool.h>

// Stub functions to be implemented with actual json-c/json.h when available
typedef struct json_object json_object;

json_object* json_object_new_object(void);
void json_object_put(json_object* obj);
json_object* json_object_new_string(const char* s);
json_object* json_object_new_int(int i);
json_object* json_object_new_double(double d);
json_object* json_object_new_boolean(bool b);
void json_object_object_add(json_object* obj, const char* key, json_object* val);
const char* json_object_to_json_string(json_object* obj);

#endif // WEBSOCKET_SERVER_STUB_H
