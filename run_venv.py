#!/bin/bash
# Script to set up and run the Resonance System in a virtual environment

set -e  # Exit on error

VENV_DIR="venv"
REQUIREMENTS="requirements.txt"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install or update requirements
if [ -f "$REQUIREMENTS" ]; then
    echo "Installing requirements..."
    pip install -r "$REQUIREMENTS"
else
    echo "Warning: $REQUIREMENTS file not found."
fi

# Run the system
echo "Starting Resonance System..."
python run_system.py "$@"

# Deactivate virtual environment when done
deactivate
