#!/bin/bash
# Build script for Kaleidoscope AI C components and bridge adapter

# Set error handling
set -e

echo "Building Kaleidoscope AI C components..."

# Create build directory if it doesn't exist
mkdir -p build

# Compile shared library
echo "Building shared library (libkaleidoscope.so)..."
make clean
make libkaleidoscope.so

# Check if build was successful
if [ ! -f "libkaleidoscope.so" ]; then
    echo "Error: Failed to build libkaleidoscope.so"
    exit 1
fi

# Install the shared library
echo "Installing shared library..."
PYTHON_BRIDGE_DIR="/home/jg/Desktop/ai_system/kaleidoscope_ai/bridge"

# Create Python bridge directory if it doesn't exist
mkdir -p "$PYTHON_BRIDGE_DIR"

# Copy to Python bridge directory
cp libkaleidoscope.so "$PYTHON_BRIDGE_DIR/"
echo "Copied to Python bridge directory: $PYTHON_BRIDGE_DIR"

# Copy to system library directory if running as root
if [ "$EUID" -eq 0 ]; then
    cp libkaleidoscope.so /usr/local/lib/
    ldconfig
    echo "Installed to system library directory: /usr/local/lib/"
else
    echo "Note: Not running as root. Skipping system-wide installation."
    echo "Run 'sudo cp libkaleidoscope.so /usr/local/lib/ && sudo ldconfig' to install system-wide."
fi

echo "Build and installation complete!"
echo "The bridge library is now available for the Python components."