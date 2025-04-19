#!/bin/bash
# Kaleidoscope AI Setup Script
# Sets up the environment for running Kaleidoscope AI

set -e  # Exit on error

echo "===== Kaleidoscope AI Setup ====="

# Check for required tools
missing_tools=()

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    echo "Please install Python 3.8 or newer and try again."
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is required but not installed."
    echo "Please install pip and try again."
    exit 1
fi

# Check for optional tools
if ! command -v r2 &> /dev/null; then
    missing_tools+=("radare2")
fi

if ! command -v ghidra_server &> /dev/null; then
    missing_tools+=("ghidra")
fi

if ! command -v retdec-decompiler &> /dev/null; then
    missing_tools+=("retdec")
fi

if ! command -v js-beautify &> /dev/null; then
    missing_tools+=("js-beautify")
fi

if [ ${#missing_tools[@]} -gt 0 ]; then
    echo "The following tools are missing and recommended for full functionality:"
    for tool in "${missing_tools[@]}"; do
        echo "- $tool"
    done
    echo "See # External Tools Setup Guide.md for installation instructions."
fi

# Create directories
echo "Creating required directories..."
mkdir -p kaleidoscope_workdir/source
mkdir -p kaleidoscope_workdir/decompiled
mkdir -p kaleidoscope_workdir/specs
mkdir -p kaleidoscope_workdir/reconstructed
mkdir -p templates
mkdir -p static/css
mkdir -p static/js
mkdir -p uploads
mkdir -p logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Default configuration if not exists
if [ ! -f config.json ]; then
    echo "Creating default configuration..."
    cat > config.json << EOF
{
  "llm_endpoint": "http://localhost:11434/api",
  "ollama_settings": {
    "default_model": "codellama",
    "available_models": ["codellama", "llama2", "mistral", "phi"],
    "temperature": 0.2,
    "max_tokens": 4096,
    "context_window": 8192
  },
  "max_workers": 4,
  "logging": {
    "level": "INFO",
    "file": "kaleidoscope.log",
    "console": true
  },
  "tools": {
    "radare2_path": "r2",
    "ghidra_path": "ghidra_server",
    "retdec_path": "retdec-decompiler",
    "js_beautify_path": "js-beautify"
  },
  "processing": {
    "chunk_size": 4096,
    "max_file_size": 100000000
  },
  "web_interface": {
    "host": "127.0.0.1",
    "port": 5000,
    "enable_upload": true,
    "max_upload_size": 50000000
  }
}
EOF
fi

# Check for Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama found. Would you like to pull the codellama model? (y/n)"
    read -r pull_model
    
    if [[ "$pull_model" == "y" || "$pull_model" == "Y" ]]; then
        echo "Pulling codellama model... This may take some time."
        ollama pull codellama
    fi
else
    echo "Ollama not found. For LLM functionality, please install Ollama from https://ollama.ai"
fi

# Make scripts executable
chmod +x kaleidoscope.py

echo "===== Setup Complete ====="
echo "You can now run Kaleidoscope AI:"
echo "- Command line: python3 kaleidoscope.py ingest --file path/to/file.exe"
echo "- Web interface: python3 kaleidoscope.py web"
echo ""
echo "For more information, see the README.md file."
