#!/bin/bash
# Build script for Kaleidoscope AI System

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Kaleidoscope AI System ===${NC}"

# Create build directory
mkdir -p build

# Determine build mode
BUILD_MODE="debug"
USE_MOCK=0

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --release)
      BUILD_MODE="release"
      ;;
    --debug)
      BUILD_MODE="debug"
      ;;
    --mock)
      USE_MOCK=1
      ;;
    --help)
      echo "Usage: $0 [--release|--debug] [--mock]"
      echo "  --release    Build in release mode (with optimizations)"
      echo "  --debug      Build in debug mode (default)"
      echo "  --mock       Use mock implementations for testing"
      exit 0
      ;;
  esac
done

# Set up compiler flags
if [ "$BUILD_MODE" == "release" ]; then
  echo -e "${GREEN}Building in RELEASE mode${NC}"
  CFLAGS="-Wall -O2 -fPIC"
else
  echo -e "${GREEN}Building in DEBUG mode${NC}"
  CFLAGS="-Wall -g -fPIC -DDEBUG"
fi

# Use mock implementation if requested
if [ "$USE_MOCK" -eq 1 ]; then
  echo -e "${YELLOW}Using MOCK implementations${NC}"
  CFLAGS="$CFLAGS -DMOCK_IMPLEMENTATION"
fi

# Source files to compile for the main executable
MAIN_SOURCE_FILES=(
    "main.c"
    "kaleidoscope_engine.c"
    "memory_graph.c"
    "pattern_recognition.c"
    "optimization.c"
    "mirrored_engine.c"
    "node_core.c"
    "data_ingestion.c"
    "self_diagnostic.c"
    "task_manager.c"
    "simulation_layer.c"
    "websocket_server.c"
    "super_node.c"
    "super_node_scripts.c"
    "ollama_integration.c"
    "quantum_cube_visualizer.c"
    "feedback_system.c"
    "learning_feedback.c"
    "web_crawler.c"
    "autonomous_learning.c"
    "system_integration.c"
    "node_dna.c"
    "supernode_integration.c"
)

# Source file for the bridge adapter
BRIDGE_SRC="bridge_adapter.c"

# Remove legacy bridge files from build
EXCLUDE_FILES=(python_to_c_bridge.c dna_bridge.c)

# All source files (including bridge for shared lib objects)
ALL_SOURCE_FILES=("${MAIN_SOURCE_FILES[@]}" "$BRIDGE_SRC")

# Compile each source file
echo -e "${GREEN}Compiling source files...${NC}"
OBJECT_FILES=()
SHARED_LIB_OBJECT_FILES=()
MAIN_EXEC_OBJECT_FILES=()

for src in "${ALL_SOURCE_FILES[@]}"; do
  if [ -f "$src" ]; then
    obj="build/$(basename "$src" .c).o"
    echo -e "${YELLOW}Compiling $src -> $obj${NC}"
    gcc $CFLAGS -I. -c "$src" -o "$obj"
    OBJECT_FILES+=("$obj") # Add to list of all objects

    # Add to shared lib objects list (exclude main.c, quantum_cube_visualizer.c)
    if [[ "$src" != "main.c" && "$src" != "quantum_cube_visualizer.c" ]]; then
        SHARED_LIB_OBJECT_FILES+=("$obj")
    fi
    # Add to main exec objects list (exclude bridge_adapter.c)
    if [[ "$src" != "$BRIDGE_SRC" ]]; then
        MAIN_EXEC_OBJECT_FILES+=("$obj")
    fi

  else
    echo -e "${RED}Warning: Source file $src not found${NC}"
  fi
done

# Link the executable
echo -e "${GREEN}Linking executable (ai_system)...${NC}"
gcc -o ai_system "${MAIN_EXEC_OBJECT_FILES[@]}" -lm -lpthread -ldl -lwebsockets -lcurl -ljansson -lglut -lGL -lGLU

# Link the shared library (libkaleidoscope.so)
echo -e "${GREEN}Building shared library (libkaleidoscope.so)...${NC}"
SHARED_LDFLAGS="-lm -lpthread -ldl -lwebsockets -lcurl -ljansson"
gcc -shared -Wl,-soname,libkaleidoscope.so \
    "${SHARED_LIB_OBJECT_FILES[@]}" \
    $SHARED_LDFLAGS \
    -o libkaleidoscope.so

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}Executable: ./ai_system${NC}"
echo -e "${YELLOW}Shared library: ./libkaleidoscope.so${NC}"

# Link shared library to Python module directory if needed
PYTHON_MODULE_DIR="kaleidoscope_ai/bridge"
if [ -d "$PYTHON_MODULE_DIR" ]; then
  echo -e "${GREEN}Copying shared library to Python module directory...${NC}"
  cp libkaleidoscope.so "$PYTHON_MODULE_DIR/"
  echo -e "${YELLOW}Library copied to $PYTHON_MODULE_DIR${NC}"
fi

echo -e "${GREEN}============================${NC}"
echo -e "${GREEN}Build is ready.${NC}"
echo -e "${YELLOW}Run using: ./run.sh${NC}"
echo -e "${GREEN}============================${NC}"
