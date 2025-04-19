#ifndef MOCK_JANSSON_H
#define MOCK_JANSSON_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Define types */
typedef struct json_t json_t;
typedef struct json_error_t {
    char text[256];
} json_error_t;

/* Define json type enums */
#define JSON_OBJECT 0
#define JSON_ARRAY 1
#define JSON_STRING 2
#define JSON_INTEGER 3
#define JSON_REAL 4
#define JSON_TRUE 5
#define JSON_FALSE 6
#define JSON_NULL 7

/* Define flags */
#define JSON_COMPACT 0

/* Make all json objects appear to be the same for mock purposes */
struct json_t {
    int type;
    char* string_value;
    int integer_value;
    double real_value;
    int ref_count;
    json_t** array_items;
    int array_size;
    json_t** object_items;
    char** object_keys;
    int object_size;
};

/* Function declarations */
json_t* json_object(void);
json_t* json_array(void);
json_t* json_string(const char* value);
json_t* json_integer(int value);
json_t* json_real(double value);
json_t* json_true(void);
json_t* json_false(void);
json_t* json_null(void);
void json_decref(json_t* json);
int json_is_string(const json_t* json);
const char* json_string_value(const json_t* json);
json_t* json_object_get(const json_t* object, const char* key);
int json_object_set_new(json_t* object, const char* key, json_t* value);
void json_array_append_new(json_t* array, json_t* value);
json_t* json_loads(const char* input, int flags, json_error_t* error);
char* json_dumps(const json_t* json, int flags);

/* Function implementations */
json_t* json_object(void) {
    json_t* obj = (json_t*)malloc(sizeof(json_t));
    memset(obj, 0, sizeof(json_t));
    obj->type = JSON_OBJECT;
    obj->ref_count = 1;
    obj->object_items = (json_t**)malloc(sizeof(json_t*) * 10);
    obj->object_keys = (char**)malloc(sizeof(char*) * 10);
    return obj;
}

json_t* json_array(void) {
    json_t* arr = (json_t*)malloc(sizeof(json_t));
    memset(arr, 0, sizeof(json_t));
    arr->type = JSON_ARRAY;
    arr->ref_count = 1;
    arr->array_items = (json_t**)malloc(sizeof(json_t*) * 10);
    return arr;
}

json_t* json_string(const char* value) {
    json_t* str = (json_t*)malloc(sizeof(json_t));
    memset(str, 0, sizeof(json_t));
    str->type = JSON_STRING;
    str->ref_count = 1;
    str->string_value = strdup(value);
    return str;
}

json_t* json_integer(int value) {
    json_t* integer = (json_t*)malloc(sizeof(json_t));
    memset(integer, 0, sizeof(json_t));
    integer->type = JSON_INTEGER;
    integer->ref_count = 1;
    integer->integer_value = value;
    return integer;
}

json_t* json_real(double value) {
    json_t* real = (json_t*)malloc(sizeof(json_t));
    memset(real, 0, sizeof(json_t));
    real->type = JSON_REAL;
    real->ref_count = 1;
    real->real_value = value;
    return real;
}

json_t* json_true(void) {
    json_t* true_val = (json_t*)malloc(sizeof(json_t));
    memset(true_val, 0, sizeof(json_t));
    true_val->type = JSON_TRUE;
    true_val->ref_count = 1;
    return true_val;
}

json_t* json_false(void) {
    json_t* false_val = (json_t*)malloc(sizeof(json_t));
    memset(false_val, 0, sizeof(json_t));
    false_val->type = JSON_FALSE;
    false_val->ref_count = 1;
    return false_val;
}

json_t* json_null(void) {
    json_t* null_val = (json_t*)malloc(sizeof(json_t));
    memset(null_val, 0, sizeof(json_t));
    null_val->type = JSON_NULL;
    null_val->ref_count = 1;
    return null_val;
}

void json_decref(json_t* json) {
    if (!json) return;
    
    json->ref_count--;
    
    if (json->ref_count <= 0) {
        if (json->type == JSON_STRING && json->string_value) {
            free(json->string_value);
        }
        
        if (json->type == JSON_ARRAY && json->array_items) {
            for (int i = 0; i < json->array_size; i++) {
                if (json->array_items[i]) {
                    json_decref(json->array_items[i]);
                }
            }
            free(json->array_items);
        }
        
        if (json->type == JSON_OBJECT) {
            if (json->object_items) {
                for (int i = 0; i < json->object_size; i++) {
                    if (json->object_items[i]) {
                        json_decref(json->object_items[i]);
                    }
                }
                free(json->object_items);
            }
            
            if (json->object_keys) {
                for (int i = 0; i < json->object_size; i++) {
                    if (json->object_keys[i]) {
                        free(json->object_keys[i]);
                    }
                }
                free(json->object_keys);
            }
        }
        
        free(json);
    }
}

int json_is_string(const json_t* json) {
    return json && json->type == JSON_STRING;
}

const char* json_string_value(const json_t* json) {
    return (json && json->type == JSON_STRING) ? json->string_value : "invalid";
}

json_t* json_object_get(const json_t* object, const char* key) {
    if (!object || object->type != JSON_OBJECT) return NULL;
    
    for (int i = 0; i < object->object_size; i++) {
        if (strcmp(object->object_keys[i], key) == 0) {
            return object->object_items[i];
        }
    }
    
    return NULL;
}

int json_object_set_new(json_t* object, const char* key, json_t* value) {
    if (!object || object->type != JSON_OBJECT || !key || !value) return -1;
    
    if (object->object_size < 10) {
        object->object_keys[object->object_size] = strdup(key);
        object->object_items[object->object_size] = value;
        object->object_size++;
        return 0;
    }
    return -1;
}

void json_array_append_new(json_t* array, json_t* value) {
    if (!array || array->type != JSON_ARRAY || !value) return;
    
    if (array->array_size < 10) {
        array->array_items[array->array_size] = value;
        array->array_size++;
    }
}

json_t* json_loads(const char* input, int flags, json_error_t* error) {
    if (!input) {
        if (error) strcpy(error->text, "Null input");
        return NULL;
    }
    
    // Simple mock that returns an object with a message
    json_t* obj = json_object();
    json_t* type_val = json_string("mock_type");
    json_object_set_new(obj, "type", type_val);
    
    json_t* message_val = json_string("This is a mock JSON parser");
    json_object_set_new(obj, "message", message_val);
    
    return obj;
}

char* json_dumps(const json_t* json, int flags) {
    if (!json) return strdup("null");
    
    switch (json->type) {
        case JSON_STRING:
            return strdup(json->string_value ? json->string_value : "null");
            
        case JSON_INTEGER: {
            char buffer[32];
            sprintf(buffer, "%d", json->integer_value);
            return strdup(buffer);
        }
        
        case JSON_REAL: {
            char buffer[64];
            sprintf(buffer, "%f", json->real_value);
            return strdup(buffer);
        }
        
        case JSON_OBJECT:
            return strdup("{\"mock\":\"object\"}");
            
        case JSON_ARRAY:
            return strdup("[\"mock\",\"array\"]");
            
        default:
            return strdup("null");
    }
}

#endif /* MOCK_JANSSON_H */