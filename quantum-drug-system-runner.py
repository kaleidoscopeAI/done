#!/usr/bin/env python3

import asyncio
import argparse
import logging
import os
import sys
import platform
from pathlib import Path
import time
import json
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quantum_drug_system.log")
    ]
)

logger = logging.getLogger("QuantumDrugSystem")

# Try to import our components (suppressing warnings)
try:
    # Import our quantum simulator components
    from quantum_drug_simulator import (
        QuantumDrugSimulator, 
        QuantumOptimizationStrategy
    )
    
    # Import visualization components
    from quantum_visualization_module import (
        MolecularVisualizer,
        QuantumMoleculeIntegrator
    )
    
    # Import Avogadro bridge
    from quantum_avogadro_bridge import (
        AvogadroBridge,
        QuantumAvogadroIntegrator
    )
    
    # Import from main program
    from quantum_drug_system_main import run_drug_discovery_pipeline
    
    components_loaded = True
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    components_loaded = False

# Check for required packages
def check_prerequisites():
    """Check if all required packages are installed"""
    required_packages = {
        "numpy": "numpy",
        "torch": "torch",
        "rdkit": "rdkit",
        "matplotlib": "matplotlib",
        "networkx": "networkx"
    }
    
    missing = []
    for name, package in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)
    
    return missing

# Check for Avogadro
def check_avogadro():
    """Check if Avogadro is installed"""
    if platform.system() == "Windows":
        # Check common installation paths on Windows
        paths = [
            r"C:\Program Files\Avogadro\bin\avogadro.exe",
            r"C:\Program Files (x86)\Avogadro\bin\avogadro.exe"
        ]
        for path in paths:
            if os.path.exists(path):
                return True
    else:
        # On Linux/Mac, check if avogadro is in the PATH
        try:
            import subprocess
            result = subprocess.run(["which", "avogadro"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            return result.returncode == 0
        except:
            pass
    
    return False

# Setup environment
def setup_environment():
    """Setup the environment for running the system"""
    if not components_loaded:
        # Check if packages are installed
        missing = check_prerequisites()
        if missing:
            logger.error(f"Missing required packages: {', '.join(missing)}")
            logger.error("Please install these packages first")
            return False
    
    # Check for Avogadro
    avogadro_installed = check_avogadro()
    if not avogadro_installed:
        logger.warning("Avogadro not found. Visualization features will be limited.")
    
    return True

# Initialize the simulator
def initialize_simulator(config=None):
    """Initialize the quantum drug simulator"""
    if not components_loaded:
        logger.error("Cannot initialize simulator: components not loaded")
        return None
    
    # Set default config if none provided
    if config is None:
        config = {
            'quantum': {
                'dimension': 16,
                'optimization_strategy': 'HYBRID'
            },
            'system': {
                'use_gpu': True
            }
        }
    
    # Map strategy string to enum
    strategy_map = {
        'VARIATIONAL': QuantumOptimizationStrategy.VARIATIONAL,
        'ADIABATIC': QuantumOptimizationStrategy.ADIABATIC,
        'HYBRID': QuantumOptimizationStrategy.HYBRID,
        'TENSOR_NETWORK': QuantumOptimizationStrategy.TENSOR_NETWORK
    }
    
    strategy = strategy_map.get(
        config['quantum'].get('optimization_strategy', 'HYBRID'),
        QuantumOptimizationStrategy.HYBRID
    )
    
    # Initialize simulator
    simulator = QuantumDrugSimulator(
        quantum_dim=config['quantum'].get('dimension', 16),
        optimization_strategy=strategy,
        enable_gpu=config['system'].get('use_gpu', True)
    )
    
    # Add default targets
    _add_default_targets(simulator)
    
    return simulator

# Add default targets
def _add_default_targets(simulator):
    """Add default protein targets"""
    default_targets = {
        "SARS-CoV-2 Spike": {
            "pdb_id": "7BZ5",
            "binding_site": [(1.2, 0.5, -0.7), (0.8, 0.3, -0.5), (1.5, 0.8, -0.9)]
        },
        "ACE2": {
            "pdb_id": "6M0J",
            "binding_site": [(2.3, 1.5, 0.7), (3.1, 2.2, 1.1), (2.8, 1.9, 0.4)]
        },
        "HDAC2": {
            "pdb_id": "4LY1",
            "binding_site": [(-1.3, 0.5, -0.7), (-1.1, 1.2, -1.1), (-1.8, 0.9, -0.4)]
        }
    }
    
    for name, data in default_targets.items():
        simulator.add_target_protein(name, data["pdb_id"], data["binding_site"])

# Load molecule from SMILES or file
def load_molecule(simulator, molecule_input, name=None):
    """Load a molecule from SMILES string or file"""
    if not simulator:
        logger.error("Cannot load molecule: simulator not initialized")
        return False
    
    # Check if input is a file
    path = Path(molecule_input)
    if path.exists() and path.is_file():
        logger.info(f"Loading molecule from file: {path}")
        
        # Determine file type by extension
        if path.suffix.lower() in ['.mol', '.sdf']:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(path))
            if mol:
                smiles = Chem.MolToSmiles(mol)
                name = name or path.stem
                return simulator.add_molecule(smiles, name=name)
            else:
                logger.error(f"Failed to load molecule from file: {path}")
                return False
        
        elif path.suffix.lower() in ['.pdb']:
            from rdkit import Chem
            mol = Chem.MolFromPDBFile(str(path))
            if mol:
                smiles = Chem.MolToSmiles(mol)
                name = name or path.stem
                return simulator.add_molecule(smiles, name=name)
            else:
                logger.error(f"Failed to load molecule from PDB file: {path}")
                return False
                
        else:
            # Assume file contains SMILES
            try:
                with open(path, 'r') as f:
                    smiles = f.read().strip()
                name = name or path.stem
                return simulator.add_molecule(smiles, name=name)
            except Exception as e:
                logger.error(f"Failed to load SMILES from file: {e}")
                return False
    else:
        # Assume input is a SMILES string
        logger.info(f"Loading molecule from SMILES: {molecule_input}")
        name = name or f"molecule_{int(time.time())}"
        return simulator.add_molecule(molecule_input, name=name)

# Run a simulation with visualization
async def run_simulation(simulator, molecule_name, args):
    """Run simulation with visualization"""
    if not simulator:
        logger.error("Cannot run simulation: simulator not initialized")
        return None
        
    if molecule_name not in simulator.molecular_database:
        logger.error(f"Molecule {molecule_name} not found in database")
        return None
    
    # Setup visualization
    if args.visualize:
        try:
            # Initialize visualizer
            visualizer = MolecularVisualizer(interactive=True)
            integrator = QuantumMoleculeIntegrator(simulator, visualizer)
            
            # Run visualization
            results = await integrator.optimize_and_visualize(
                molecule_name, 
                steps=args.steps,
                save_images=args.save_images
            )
            
            return results
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            # Fall back to non-visual simulation
    
    # Run non-visual simulation
    return await run_drug_discovery_pipeline(simulator, molecule_name, args.target)

# Launch Avogadro visualization
async def launch_avogadro_visualization(simulator, molecule_name):
    """Launch Avogadro with the current molecule and quantum field"""
    if not simulator:
        logger.error("Cannot launch Avogadro: simulator not initialized")
        return False
        
    if molecule_name not in simulator.molecular_database:
        logger.error(f"Molecule {molecule_name} not found in database")
        return False
    
    try:
        # Initialize Avogadro bridge
        avogadro_integrator = QuantumAvogadroIntegrator(simulator)
        
        # Launch visualization
        return await avogadro_integrator.visualize_molecule(molecule_name)
    except Exception as e:
        logger.error(f"Avogadro launch error: {e}")
        return False

# Save results
def save_results(results, output_path):
    """Save simulation results to file"""
    if not results:
        logger.error("No results to save")
        return False
    
    try:
        # Create directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert results to JSON-friendly format
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if hasattr(v, 'tolist'):  # Convert numpy arrays
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, (int, float, str, bool, list, dict)):
                        json_results[key][k] = v
                    else:
                        json_results[key][k] = str(v)
            elif isinstance(value, (int, float, str, bool)):
                json_results[key] = value
            elif hasattr(value, 'tolist'):  # Convert numpy arrays
                json_results[key] = value.tolist()
            else:
                json_results[key] = str(value)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

# Parse arguments
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quantum Drug Discovery System"
    )
    
    # Input options
    parser.add_argument("--smiles", type=str, help="Input molecule SMILES string")
    parser.add_argument("--file", type=str, help="Input molecule file path")
    parser.add_argument("--name", type=str, help="Name for the molecule")
    
    # Simulation options
    parser.add_argument("--target", type=str, help="Target protein name")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--avogadro", action="store_true", help="Launch Avogadro visualization")
    parser.add_argument("--save-images", action="store_true", help="Save visualization images")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file path for results")
    
    # System options
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    return parser.parse_args()

# Main function
async def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Check environment
    if not setup_environment():
        return 1
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    if config is None:
        config = {
            'quantum': {
                'dimension': 16,
                'optimization_strategy': 'HYBRID'
            },
            'system': {
                'use_gpu': args.gpu
            }
        }
    
    # Initialize simulator
    simulator = initialize_simulator(config)
    if not simulator:
        return 1
    
    # Handle input molecule
    molecule_name = None
    
    if args.smiles:
        if load_molecule(simulator, args.smiles, args.name):
            molecule_name = args.name or f"molecule_{int(time.time())}"
            logger.info(f"Loaded molecule from SMILES as {molecule_name}")
        else:
            logger.error("Failed to load molecule from SMILES")
            return 1
    elif args.file:
        if load_molecule(simulator, args.file, args.name):
            molecule_name = args.name or Path(args.file).stem
            logger.info(f"Loaded molecule from file as {molecule_name}")
        else:
            logger.error("Failed to load molecule from file")
            return 1
    else:
        # Default to caffeine if no molecule specified
        logger.info("No molecule specified, using caffeine as example")
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        if load_molecule(simulator, caffeine_smiles, "caffeine"):
            molecule_name = "caffeine"
        else:
            logger.error("Failed to load default molecule")
            return 1
    
    # Run simulation
    logger.info(f"Running simulation for {molecule_name}")
    results = await run_simulation(simulator, molecule_name, args)
    
    # Launch Avogadro if requested
    if args.avogadro and results:
        logger.info("Launching Avogadro visualization")
        await launch_avogadro_visualization(simulator, molecule_name)
    
    # Save results if requested
    if args.output and results:
        save_results(results, args.output)
    
    logger.info("Simulation completed successfully")
    return 0

# Entry point
if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
