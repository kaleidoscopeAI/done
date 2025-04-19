#!/bin/bash
# Script to perform final setup steps for Quantum Kaleidoscope (Robust Error Handling v2)

# --- !! VERIFY THESE PATHS !! ---
# Base directory for your project
PROJECT_DIR="$HOME/Music/kaleidoscope_project"
# Directory containing the original code files (renamed to .py)
SOURCE_DIR="$HOME/Music/quatumunravel"
# Target directory for the integrated system (will be created by integration script)
INTEGRATED_DIR="$PROJECT_DIR/integrated_system"
# Path to the Python virtual environment
VENV_DIR="$PROJECT_DIR/venv"
# Directory for building external dependencies like llama.cpp
LLAMA_CPP_BUILD_DIR="$HOME/.cache/unravel-ai/build"
# --- End Editable Paths ---

# Exit immediately if a command exits with a non-zero status.
set -e

# Function for logging errors and exiting
error_exit() {
    echo "ERROR: $1" >&2
    # Optional: Add cleanup logic here if needed
    exit 1
}

# Trap script exit to report failure location
trap 'echo "--- SCRIPT FAILED AT LINE $LINENO ---" >&2' ERR

echo "--- Starting Final Kaleidoscope Setup ---"
echo ">>> Using Project Directory: $PROJECT_DIR"
echo ">>> Using Source Directory: $SOURCE_DIR"

# --- 1. Ensure Project Directory Exists & Navigate ---
echo ">>> Ensuring Project Directory Exists: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR" || error_exit "Failed to create project directory $PROJECT_DIR"
cd "$PROJECT_DIR" || error_exit "Failed to navigate into project directory $PROJECT_DIR"
echo "    Current directory: $(pwd)"

# --- 2. Create/Activate Virtual Environment ---
echo ">>> Activating Virtual Environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    echo "    Virtual environment not found, creating..."
    python3 -m venv venv || error_exit "Failed to create virtual environment."
fi

ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if [ -f "$ACTIVATE_SCRIPT" ]; then
    # Use '.' which is POSIX compliant for sourcing scripts
    # Ensure this line is clean in the source
    . "$ACTIVATE_SCRIPT" || error_exit "Failed to source virtual environment activation script."
    echo "    Virtual environment activated."
else
    error_exit "Virtual environment activation script not found at $ACTIVATE_SCRIPT"
fi

# --- 3. Install/Verify System Dependencies ---
echo ">>> Verifying System Dependencies (cmake, git, C++ compiler)..."
command -v cmake >/dev/null 2>&1 || error_exit "cmake is required but not found. Please install it."
command -v git >/dev/null 2>&1 || error_exit "git is required but not found. Please install it."
command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || error_exit "C++ compiler (g++ or clang++) is required but not found."
echo "    System dependencies verified."

# --- 4. Install Python Dependencies (Corrected) ---
echo ">>> Installing Python Dependencies..."
pip install --upgrade pip || error_exit "Failed to upgrade pip."
# Removed d3, added error checking
pip install numpy torch websockets fastapi uvicorn flask flask-socketio requests networkx matplotlib scipy pennylane plotly paramiko docker kubernetes streamlit transformers huggingface_hub llama-cpp-python tokenizers ctransformers spacy colorlog eventlet pandas Pillow psutil || error_exit "Failed to install Python dependencies."
echo "    Python dependencies installed."

# --- 5. Build and Install llama.cpp (if needed) ---
echo ">>> Building/Installing llama.cpp (if needed)..."
mkdir -p "$LLAMA_CPP_BUILD_DIR" || error_exit "Failed to create llama.cpp build parent directory."
cd "$LLAMA_CPP_BUILD_DIR" || error_exit "Failed to cd into llama.cpp build directory."

if [ ! -d "llama.cpp" ]; then
    echo "    Cloning llama.cpp repository..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git || error_exit "Failed to clone llama.cpp."
else
    echo "    llama.cpp repository already exists."
fi

LLAMA_CPP_SRC_DIR="$LLAMA_CPP_BUILD_DIR/llama.cpp"
cd "$LLAMA_CPP_SRC_DIR" || error_exit "Failed to cd into llama.cpp source directory."
mkdir -p build
cd build || error_exit "Failed to cd into llama.cpp build subdirectory."

echo "    Configuring llama.cpp with CMake..."
# Add CMAKE_ARGS if needed, e.g., CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
cmake .. $CMAKE_ARGS || error_exit "cmake configuration failed."

echo "    Building llama.cpp..."
# Adjust thread count '-j' if necessary
cmake --build . --config Release -j $(nproc) || error_exit "cmake build failed."

echo "    Installing llama-server to $HOME/.local/bin/ ..."
INSTALL_BIN_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_BIN_DIR" || echo "WARN: Failed to create $INSTALL_BIN_DIR/"
LLAMA_SERVER_BIN="bin/llama-server"
if [ -f "$LLAMA_SERVER_BIN" ]; then
    cp -f "$LLAMA_SERVER_BIN" "$INSTALL_BIN_DIR/" || echo "WARN: Failed to copy llama-server to $INSTALL_BIN_DIR. Ensure it's runnable or adjust PATH."
else
    echo "WARN: llama-server binary not found after build in $(pwd)/bin/"
fi
echo "    IMPORTANT: Ensure $INSTALL_BIN_DIR is in your PATH environment variable."
if [[ ":$PATH:" != *":${INSTALL_BIN_DIR}:"* ]]; then
    echo "    WARNING: $INSTALL_BIN_DIR does not seem to be in your PATH."
fi
echo "    llama.cpp build/install complete."

# --- 6. Run Deployment Fix Script ---
echo ">>> Running Deployment Fix Script..."
cd "$PROJECT_DIR" || error_exit "Failed to cd back to project directory for fix script."
# Ensure the source script path is correct
FIX_SCRIPT_SOURCE="$SOURCE_DIR/deployment-fix.py"
DEPLOY_SCRIPT_SOURCE="$SOURCE_DIR/deployment-script.py"
[ ! -f "$FIX_SCRIPT_SOURCE" ] && error_exit "Fix script not found at $FIX_SCRIPT_SOURCE"
[ ! -f "$DEPLOY_SCRIPT_SOURCE" ] && error_exit "Deployment script not found at $DEPLOY_SCRIPT_SOURCE"
# Ensure deployment-fix.py has the correction from the previous step
python "$FIX_SCRIPT_SOURCE" --script "$DEPLOY_SCRIPT_SOURCE" || error_exit "Deployment fix script failed."
echo "    Deployment fix script executed."

# --- 7. Run Integration Script ---
echo ">>> Running Integration Script (integrate_kaleidoscope.py)..."
cd "$PROJECT_DIR" || error_exit "Failed to cd back to project directory for integration."
# Ensure the source script path is correct
INTEGRATION_SCRIPT_SOURCE="$SOURCE_DIR/integrate_kaleidoscope.py"
[ ! -f "$INTEGRATION_SCRIPT_SOURCE" ] && error_exit "Integration script not found at $INTEGRATION_SCRIPT_SOURCE"
[ ! -d "$SOURCE_DIR" ] && error_exit "Source directory $SOURCE_DIR not found for integration."
# Using integrate_kaleidoscope.py as the example
python "$INTEGRATION_SCRIPT_SOURCE" "$SOURCE_DIR" "$INTEGRATED_DIR" --force || error_exit "Integration script failed."
echo "    Integration script executed. Check '$INTEGRATED_DIR' for results."

# --- 8. Post-Script Manual Steps ---
echo ""
echo "--- ✅ Automated Steps Complete ---"
echo ""
echo "--- ⚠️ MANUAL ACTIONS REQUIRED ---"
echo "1.  Navigate to the integrated system directory:"
echo "    cd \"$INTEGRATED_DIR\""
echo "2.  Create/Edit necessary configuration files (e.g., config.json, llm_config.json)."
echo "    Ensure LLM model paths and any API/SSH keys are correctly set relative to '$INTEGRATED_DIR'."
echo "3.  Download the required LLM GGUF model file (if you haven't already)."
echo "    Place it in the location specified in your configuration."
echo "    Example: huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ~/.cache/unravel-ai/models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF --local-dir-use-symlinks False"
echo "4.  If using the 'llamacpp_api' provider, START the llama.cpp server in a SEPARATE TERMINAL before proceeding:"
echo "    Example: llama-server -m <path_to_your_model.gguf> -c 2048 --host 127.0.0.1 --port 8080"
echo "5.  Once configured and the LLM server is running (if needed), RUN the launcher FROM THE '$INTEGRATED_DIR' directory:"
echo "    python quantum_kaleidoscope_launcher.py"
echo "--- End of Script ---"

# Deactivate virtual environment (optional)
# deactivate

echo "Setup script finished."
