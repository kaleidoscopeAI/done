# Kaleidoscope Quantum Cube
## Advanced Drug Discovery System

The Kaleidoscope Quantum Cube is a groundbreaking framework for drug discovery and molecular modeling that combines tensor mathematics, dynamic string networks, and quantum-inspired visualization to discover and optimize potential drug candidates.

## Core Features

1. **Tensor-Based Biological Modeling**
   - Multi-dimensional data representation
   - Advanced tensor decomposition (CP, Tucker)
   - Latent pattern discovery in complex biological data

2. **Dynamic String Networks**
   - Physics-based molecular interaction modeling
   - Stress-responsive connections that adapt to binding
   - Hooke's law + biological stress gradients

3. **Quantum Visualization**
   - 3D interactive visualization of molecular dynamics
   - Binding pocket exploration
   - Tensor decomposition visualization

4. **Advanced Drug Discovery Pipeline**
   - Binding site identification
   - Molecular docking
   - Drug-likeness prediction
   - ADMET property prediction
   - Lead compound identification

## Installation

### Prerequisites

- Python 3.9+ 
- NumPy, SciPy, NetworkX, matplotlib
- scikit-learn, pandas
- Optional: PyQt5 for interactive visualization, RDKit for advanced cheminformatics

### Automatic Setup

The script includes automatic virtual environment setup. Simply run:

```bash
python kaleidoscope_quantum_cube.py
```

The script will:
1. Check if running in a virtual environment
2. Create a virtual environment if needed
3. Install all required dependencies
4. Run itself in the new environment

## Usage

### Demo Mode

Run the built-in demo to see the system in action:

```bash
python kaleidoscope_quantum_cube.py --mode demo
```

This will:
- Create a mock receptor and ligand
- Identify binding sites
- Perform molecular docking
- Evaluate drug-likeness and ADMET properties
- Generate visualizations

### Batch Mode

For processing real data:

```bash
python kaleidoscope_quantum_cube.py --mode batch --receptor path/to/receptor.pdb --ligand path/to/ligand.mol --output results_dir
```

### Command-Line Options

- `--mode`: Choose between `demo`, `batch`, or `interactive` (default: `demo`)
- `--receptor`: Path to receptor PDB file (required for batch mode)
- `--ligand`: Path to ligand file (required for batch mode)
- `--output`: Directory for saving results (default: `results`)

## Advanced Usage

### Customizing Parameters

You can modify physics parameters in the code:
- String elastic constants (`k`)
- Stress sensitivity (`alpha`)
- Binding interaction thresholds

### Extending Functionality

The system's modular design allows for:
- Adding custom tensor dimensions
- Creating new stress field types
- Implementing advanced ADMET prediction models
- Adding your own visualization capabilities

## Mathematical Foundation

The system integrates:
1. Tensor decomposition for high-dimensional data analysis
2. Dynamic string physics modeling biological interactions
3. Stress-based adaptation for environment-responsive simulation
4. Multi-scale clustering for emergent pattern detection