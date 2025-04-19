#!/bin/bash
# Build and test script for the AI system

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building AI System ===${NC}"

# Create build directory if it doesn't exist
mkdir -p build

# Compile all source files individually
echo -e "${GREEN}Compiling source files...${NC}"

# List of source files (excluding bridge_adapter.c which is compiled separately)
SOURCE_FILES=(
    "kaleidoscope_engine.c"
    "memory_graph.c"
    "pattern_recognition.c"
    "optimization.c"
    "mirrored_engine.c" 
    "node_core.c"
    "data_ingestion.c"
    "simulation_layer.c"
    "task_manager.c"
    "self_diagnostic.c"
    "websocket_server.c"
    "main.c"
)

# Compile each source file
for src in "${SOURCE_FILES[@]}"; do
    obj="build/$(basename $src .c).o"
    echo -e "${YELLOW}Compiling $src -> $obj${NC}"
    gcc -c -Wall -g -O2 -fPIC -I. "$src" -o "$obj"
done

# Compile bridge adapter separately
echo -e "${YELLOW}Compiling bridge_adapter.c${NC}"
gcc -c -Wall -g -O2 -fPIC -I. "bridge_adapter.c" -o "build/bridge_adapter.o"

# Link the executable
echo -e "${GREEN}Linking executable...${NC}"
gcc -Wall -g build/*.o -o ai_system -lm

# Link shared library (libkaleidoscope.so)
echo -e "${GREEN}Building shared library...${NC}"
gcc -shared -Wl,-soname,libkaleidoscope.so build/bridge_adapter.o build/memory_graph.o \
    build/kaleidoscope_engine.o build/pattern_recognition.o build/optimization.o \
    build/mirrored_engine.o build/node_core.o build/data_ingestion.o \
    build/simulation_layer.o build/task_manager.o build/self_diagnostic.o \
    -o libkaleidoscope.so

# Run tests
echo -e "${GREEN}Running tests...${NC}"
MOCK_IMPLEMENTATION=1 ./ai_system test

echo -e "${GREEN}Build and test complete!${NC}"
