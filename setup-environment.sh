#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project name
PROJECT_NAME="quantum_drug_discovery"

# Banner
echo -e "${BLUE}"
echo "======================================================================"
echo "    Quantum Drug Discovery System - Environment Setup"
echo "======================================================================"
echo -e "${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if we're running on Ubuntu
is_ubuntu() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            return 0
        fi
    fi
    return 1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PY_VERSION=$(python3 --version | cut -d ' ' -f 2)
        PY_MAJOR=$(echo $PY_VERSION | cut -d '.' -f 1)
        PY_MINOR=$(echo $PY_VERSION | cut -d '.' -f 2)
        
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
            echo -e "${GREEN}Python $PY_VERSION detected, which is compatible.${NC}"
            return 0
        else
            echo -e "${RED}Python $PY_VERSION detected, but version 3.8+ is required.${NC}"
            return 1
        fi
    else
        echo -e "${RED}Python 3 not found. Please install Python 3.8 or newer.${NC}"
        return 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    echo -e "${BLUE}Setting up Python virtual environment...${NC}"
    
    if command_exists python3; then
        # Create virtual environment
        python3 -m venv .venv
        
        # Activate virtual environment
        source .venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        echo -e "${GREEN}Virtual environment created and activated.${NC}"
        return 0
    else
        echo -e "${RED}Failed to create virtual environment: Python 3 not found.${NC}"
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    echo -e "${BLUE}Installing system dependencies...${NC}"
    
    if is_ubuntu; then
        echo -e "${BLUE}Ubuntu detected, using apt for system packages.${NC}"
        
        # Install build tools and libraries
        echo -e "${YELLOW}Installing build dependencies...${NC}"
        sudo apt-get update
        sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev curl \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
            libffi-dev liblzma-dev python3-dev python3-pip git
            
        # Install Avogadro if available
        echo -e "${YELLOW}Checking for Avogadro package...${NC}"
        if apt-cache show avogadro >/dev/null 2>&1; then
            echo -e "${YELLOW}Installing Avogadro...${NC}"
            sudo apt-get install -y avogadro
        else
            echo -e "${YELLOW}Avogadro package not found in repositories. Will try alternative installation.${NC}"
            # Try alternative installation methods
            if command_exists snap; then
                echo -e "${YELLOW}Trying to install Avogadro via snap...${NC}"
                sudo snap install avogadro
            else
                echo -e "${YELLOW}Installing Avogadro from source would be required. This is not automated.${NC}"
                echo -e "${YELLOW}You can manually install it later from https://avogadro.cc/${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Non-Ubuntu system detected. You may need to manually install dependencies.${NC}"
        echo -e "${YELLOW}Required system packages: build tools, OpenGL libraries, Qt libraries${NC}"
        
        # Check for macOS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo -e "${BLUE}macOS detected, checking for Homebrew...${NC}"
            if command_exists brew; then
                echo -e "${YELLOW}Installing dependencies via Homebrew...${NC}"
                brew install cmake qt5 open-babel
                
                # Check for Avogadro
                if brew list --formula | grep -q avogadro; then
                    echo -e "${GREEN}Avogadro is already installed.${NC}"
                else
                    echo -e "${YELLOW}Installing Avogadro...${NC}"
                    brew install avogadro
                fi
            else
                echo -e "${YELLOW}Homebrew not found. Please install Homebrew and try again.${NC}"
                echo -e "${YELLOW}Visit https://brew.sh for installation instructions.${NC}"
            fi
        fi
    fi
    
    echo -e "${GREEN}System dependencies installed.${NC}"
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    # Basic dependencies
    pip install numpy scipy matplotlib networkx
    
    # ML/Quantum dependencies
    pip install torch
    
    # Chemistry dependencies
    pip install rdkit-pypi
    
    # Visualization dependencies
    pip install py3Dmol
    
    echo -e "${GREEN}Python dependencies installed.${NC}"
}

# Function to set up the project directory
setup_project() {
    echo -e "${BLUE}Setting up project structure...${NC}"
    
    # Create project directories
    mkdir -p $PROJECT_NAME/data
    mkdir -p $PROJECT_NAME/visualizations
    mkdir -p $PROJECT_NAME/results
    
    # Create convenience scripts
    
    # Activation script
    cat > activate.sh << 'EOL'
#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
EOL
    chmod +x activate.sh
    
    # Run script
    cat > run.sh << 'EOL'
#!/bin/bash
source ./activate.sh
python quantum_drug_system_runner.py "$@"
EOL
    chmod +x run.sh
    
    # Create a default configuration file
    mkdir -p config
    cat > config/default_config.json << 'EOL'
{
    "quantum": {
        "dimension": 16,
        "optimization_strategy": "HYBRID"
    },
    "system": {
        "use_gpu": true,
        "threads": 4
    },
    "visualization": {
        "enabled": true,
        "update_interval": 0.1,
        "save_path": "visualizations"
    }
}
EOL
    
    echo -e "${GREEN}Project structure created.${NC}"
}

# Function to install source files
install_source_files() {
    echo -e "${BLUE}Installing quantum drug discovery source files...${NC}"
    
    # Check if source files exist in the current directory
    if [ -f "quantum_drug_simulator.py" ] && \
       [ -f "quantum_visualization_module.py" ] && \
       [ -f "quantum_avogadro_bridge.py" ] && \
       [ -f "quantum_drug_system_runner.py" ]; then
        
        # Copy files to project directory
        cp quantum_drug_simulator.py $PROJECT_NAME/
        cp quantum_visualization_module.py $PROJECT_NAME/
        cp quantum_avogadro_bridge.py $PROJECT_NAME/
        cp quantum_drug_system_main.py $PROJECT_NAME/
        cp quantum_drug_system_runner.py $PROJECT_NAME/
        
        echo -e "${GREEN}Source files installed.${NC}"
    else
        echo -e "${RED}Source files not found. Please make sure all required files are in the current directory.${NC}"
        echo -e "${YELLOW}Required files:${NC}"
        echo -e "${YELLOW}- quantum_drug_simulator.py${NC}"
        echo -e "${YELLOW}- quantum_visualization_module.py${NC}"
        echo -e "${YELLOW}- quantum_avogadro_bridge.py${NC}"
        echo -e "${YELLOW}- quantum_drug_system_main.py${NC}"
        echo -e "${YELLOW}- quantum_drug_system_runner.py${NC}"
        return 1
    fi
}

# Function to run a basic test
run_test() {
    echo -e "${BLUE}Running a basic test...${NC}"
    
    cd $PROJECT_NAME
    
    # Activate the environment
    source ../.venv/bin/activate
    
    # Run a simple test
    echo -e "${YELLOW}Testing system initialization...${NC}"
    python -c "
import os
import sys
try:
    from quantum_drug_simulator import QuantumDrugSimulator, QuantumOptimizationStrategy
    simulator = QuantumDrugSimulator()
    simulator.add_molecule('CCO', 'ethanol')
    print('Simulator test successful!')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
    
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}Basic test successful!${NC}"
    else
        echo -e "${RED}Test failed. Please check the error messages above.${NC}"
    fi
    
    cd ..
    
    return $TEST_RESULT
}

# Function to print final instructions
print_instructions() {
    echo -e "${BLUE}====================== SETUP COMPLETE =======================${NC}"
    echo -e "${GREEN}The Quantum Drug Discovery System has been set up successfully!${NC}"
    echo -e "${YELLOW}To use the system:${NC}"
    echo -e "${YELLOW}1. Activate the environment:${NC}"
    echo -e "   ${BLUE}source activate.sh${NC}"
    echo -e "${YELLOW}2. Run the system:${NC}"
    echo -e "   ${BLUE}./run.sh --smiles CCO --visualize${NC}"
    echo -e "${YELLOW}For more options:${NC}"
    echo -e "   ${BLUE}./run.sh --help${NC}"
    echo -e "${YELLOW}The system is configured with default settings in:${NC}"
    echo -e "   ${BLUE}config/default_config.json${NC}"
    echo -e "${BLUE}=============================================================${NC}"
}

# Main setup process
echo -e "${BLUE}Starting setup process...${NC}"

# Check Python
if ! check_python; then
    echo -e "${RED}Setup failed: Python requirements not met.${NC}"
    exit 1
fi

# Install system dependencies
install_system_deps

# Setup virtual environment
if ! setup_venv; then
    echo -e "${RED}Setup failed: Could not create virtual environment.${NC}"
    exit 1
fi

# Install Python dependencies
install_python_deps

# Setup project directory
setup_project

# Install source files
if ! install_source_files; then
    echo -e "${RED}Setup failed: Could not install source files.${NC}"
    exit 1
fi

# Run test
if ! run_test; then
    echo -e "${RED}Setup completed but tests failed. The system may not work correctly.${NC}"
else
    # Print instructions
    print_instructions
fi

# Deactivate virtual environment
deactivate

echo -e "${BLUE}Setup process completed.${NC}"
