#!/bin/bash
# Quantum Kaleidoscope Setup Script
# This script sets up and runs the Quantum Kaleidoscope system

set -e  # Exit on error

# Configuration
INSTALL_DIR="quantum_kaleidoscope"
PORT=8000
DIMENSION=128
AUTO_GEN=true
PYTHON_CMD="python3"  # Change to "python" if needed for your system

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "========================================================"
echo "      Quantum Kaleidoscope Installation Script"
echo "========================================================"
echo -e "${NC}"

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/static"
mkdir -p "$INSTALL_DIR/data"
cd "$INSTALL_DIR"

# Check Python and required libraries
echo -e "${GREEN}Checking Python installation...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python 3.6 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}Installing required Python packages...${NC}"
$PYTHON_CMD -m pip install numpy flask flask-cors flask-socketio sentence-transformers --quiet

# Create the core system file
echo -e "${GREEN}Creating Quantum Kaleidoscope core system...${NC}"
cat > quantum_kaleidoscope.py << 'EOL'
#!/usr/bin/env python3
"""
Quantum Kaleidoscope: Integrated Cognitive Intelligence System
=============================================================

Core engine with quantum-inspired processing, advanced pattern recognition,
and multidimensional data analysis capabilities.

This system combines:
- Quantum string theory principles for tensor field processing
- SuperNode architecture for distributed cognition
- Kaleidoscope pattern recognition for insight generation
- Hyperdimensional computing for emergent understanding
"""

import os
import sys
import time
import uuid
import json
import math
import logging
import hashlib
import threading
import re
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuantumKaleidoscope")

###########################################
# Core Data Structures
###########################################

class ResonanceMode(Enum):
    """Modes of resonance for processing"""
    LINEAR = auto()
    NONLINEAR = auto()
    QUANTUM = auto()
    CHAOTIC = auto()
    TOPOLOGICAL = auto()
    HYBRID = auto()

@dataclass
class SuperNodeDNA:
    """DNA representation for SuperNode"""
    # Core DNA parameters encoded as vector
    encoding: np.ndarray
    
    # Processing mode weights
    linear_weight: float = 0.2
    nonlinear_weight: float = 0.3
    quantum_weight: float = 0.1
    chaotic_weight: float = 0.1
    topological_weight: float = 0.3
    
    # Evolution parameters
    mutation_rate: float = 0.005
    crossover_points: int = 3
    
    # Structural parameters
    connectivity_pattern: np.ndarray = field(default_factory=lambda: np.random.rand(64))
    activation_thresholds: np.ndarray = field(default_factory=lambda: np.random.rand(16) * 0.5 + 0.3)
    
    def evolve(self) -> 'SuperNodeDNA':
        """Evolve DNA through mutation and potential crossover"""
        new_dna = SuperNodeDNA(
            encoding=self.encoding.copy(),
            linear_weight=self.linear_weight,
            nonlinear_weight=self.nonlinear_weight,
            quantum_weight=self.quantum_weight,
            chaotic_weight=self.chaotic_weight,
            topological_weight=self.topological_weight,
            mutation_rate=self.mutation_rate,
            crossover_points=self.crossover_points,
            connectivity_pattern=self.connectivity_pattern.copy(),
            activation_thresholds=self.activation_thresholds.copy()
        )
        
        # Apply mutations
        # Encoding mutation
        mask = np.random.random(new_dna.encoding.size) < self.mutation_rate
        mutation = np.random.normal(0, 0.1, new_dna.encoding.size)
        new_dna.encoding[mask] += mutation[mask]
        norm = np.linalg.norm(new_dna.encoding)
        if norm > 0:
            new_dna.encoding /= norm
            
        # Weight mutations
        if np.random.random() < self.mutation_rate * 5:  # Higher chance to mutate weights
            delta = np.random.normal(0, 0.05)
            new_dna.linear_weight = np.clip(new_dna.linear_weight + delta, 0.05, 0.5)
            
        if np.random.random() < self.mutation_rate * 5:
            delta = np.random.normal(0, 0.05)
            new_dna.nonlinear_weight = np.clip(new_dna.nonlinear_weight + delta, 0.05, 0.5)
        
        if np.random.random() < self.mutation_rate * 5:
            delta = np.random.normal(0, 0.02)
            new_dna.quantum_weight = np.clip(new_dna.quantum_weight + delta, 0.02, 0.3)
        
        if np.random.random() < self.mutation_rate * 5:
            delta = np.random.normal(0, 0.02)
            new_dna.chaotic_weight = np.clip(new_dna.chaotic_weight + delta, 0.02, 0.3)
            
        # Normalize weights to sum to 1.0
        total = (new_dna.linear_weight + new_dna.nonlinear_weight + 
                new_dna.quantum_weight + new_dna.chaotic_weight + 
                new_dna.topological_weight)
        
        new_dna.linear_weight /= total
        new_dna.nonlinear_weight /= total
        new_dna.quantum_weight /= total
        new_dna.chaotic_weight /= total
        new_dna.topological_weight /= total
        
        # Connectivity pattern mutation
        mask = np.random.random(new_dna.connectivity_pattern.size) < self.mutation_rate
        mutation = np.random.normal(0, 0.1, new_dna.connectivity_pattern.size)
        new_dna.connectivity_pattern[mask] += mutation[mask]
        new_dna.connectivity_pattern = np.clip(new_dna.connectivity_pattern, 0, 1)
        
        # Activation thresholds mutation
        mask = np.random.random(new_dna.activation_thresholds.size) < self.mutation_rate
        mutation = np.random.normal(0, 0.05, new_dna.activation_thresholds.size)
        new_dna.activation_thresholds[mask] += mutation[mask]
        new_dna.activation_thresholds = np.clip(new_dna.activation_thresholds, 0.1, 0.9)
        
        return new_dna
    
    @staticmethod
    def generate(dimension: int) -> 'SuperNodeDNA':
        """Generate a new random DNA with the specified dimension"""
        encoding = np.random.randn(dimension)
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding /= norm
        
        return SuperNodeDNA(encoding=encoding)

@dataclass
class SuperNodeState:
    """State representation for SuperNode"""
    # Current state vector
    current: np.ndarray
    
    # Memory buffer for recent states
    memory: deque = field(default_factory=lambda: deque(maxlen=256))
    
    # Attractor states (stable patterns)
    attractors: List[np.ndarray] = field(default_factory=list)
    
    # Energy and stability metrics
    energy: float = 1.0
    stability: float = 0.0
    coherence: float = 0.0
    
    # Processing history
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Timestamps
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    def update(self, new_state: np.ndarray) -> None:
        """Update the state with a new state vector"""
        # Add current state to memory
        self.memory.append(self.current.copy())
        
        # Compute stability (similarity between new state and current)
        norm_current = np.linalg.norm(self.current)
        norm_new = np.linalg.norm(new_state)
        
        if norm_current > 1e-9 and norm_new > 1e-9:
            similarity = np.dot(self.current, new_state) / (norm_current * norm_new)
            self.stability = max(0.0, similarity)  # Cosine similarity as stability
        else:
            self.stability = 0.0
        
        # Update current state
        self.current = new_state.copy()
        
        # Compute coherence (average similarity among recent states)
        if len(self.memory) >= 5:
            recent_states = list(self.memory)[-5:]
            similarities = []
            
            for i in range(len(recent_states)):
                for j in range(i+1, len(recent_states)):
                    state_i = recent_states[i]
                    state_j = recent_states[j]
                    
                    norm_i = np.linalg.norm(state_i)
                    norm_j = np.linalg.norm(state_j)
                    
                    if norm_i > 1e-9 and norm_j > 1e-9:
                        sim = np.dot(state_i, state_j) / (norm_i * norm_j)
                        similarities.append(max(0.0, sim))
            
            if similarities:
                self.coherence = np.mean(similarities)
        
        # Update energy based on stability
        if self.stability < 0.3:
            # Low stability increases energy (exploring)
            self.energy = min(1.0, self.energy + 0.05)
        elif self.stability > 0.8:
            # High stability decreases energy (stabilizing)
            self.energy = max(0.1, self.energy - 0.02)
        
        # Record history
        self.history.append({
            "timestamp": time.time(),
            "stability": self.stability,
            "coherence": self.coherence,
            "energy": self.energy
        })
        
        self.last_update = time.time()
        
        # Check for attractor states
        self._check_attractor()
    
    def add_attractor(self, attractor: np.ndarray) -> None:
        """Add a stable attractor state"""
        # Check if similar attractor already exists
        for existing_attractor in self.attractors:
            norm_existing = np.linalg.norm(existing_attractor)
            norm_new = np.linalg.norm(attractor)
            
            if norm_existing > 1e-9 and norm_new > 1e-9:
                similarity = np.dot(existing_attractor, attractor) / (norm_existing * norm_new)
                if similarity > 0.9:
                    # Very similar to existing attractor, don't add
                    return
        
        # Add new attractor
        self.attractors.append(attractor.copy())
        
        # Limit number of attractors
        if len(self.attractors) > 10:
            self.attractors.pop(0)  # Remove oldest
    
    def _check_attractor(self) -> None:
        """Check if current state is a potential attractor"""
        # A state is an attractor if it's very stable over time
        if len(self.history) < 5:
            return
        
        # Get recent stability values
        recent_stability = [entry["stability"] for entry in list(self.history)[-5:]]
        avg_stability = np.mean(recent_stability)
        
        # If consistently stable, consider as attractor
        if avg_stability > 0.85 and self.coherence > 0.7:
            self.add_attractor(self.current)

@dataclass
class Pattern:
    """Pattern detected in data"""
    id: str
    type: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    
    def similarity(self, other: 'Pattern') -> float:
        """Calculate similarity with another pattern"""
        if self.vector.shape != other.vector.shape:
            return 0.0
        
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        
        if norm_self < 1e-9 or norm_other < 1e-9:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(self.vector, other.vector) / (norm_self * norm_other)
        return max(0.0, similarity)

@dataclass
class Insight:
    """Insight derived from patterns"""
    id: str
    type: str
    patterns: List[str]  # Pattern IDs
    vector: np.ndarray   # Vector representation
    description: str
    confidence: float = 0.5
    importance: float = 0.5
    novelty: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Perspective:
    """Integrated view combining multiple insights"""
    id: str
    insight_ids: List[str]
    vector: np.ndarray
    strength: float
    coherence: float
    novelty: float
    impact: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = asdict(self)
        # Convert numpy array to list for JSON serialization
        result['vector'] = self.vector.tolist()
        return result

@dataclass
class QuantumState:
    """Quantum state representation"""
    num_qubits: int
    # Use sparse representation for better memory efficiency
    amplitudes: Dict[int, complex] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with |0> state if empty"""
        if not self.amplitudes:
            self.amplitudes[0] = 1.0 + 0j
    
    def apply_hadamard(self, target: int):
        """Apply Hadamard gate to target qubit"""
        if target >= self.num_qubits:
            raise ValueError(f"Target qubit {target} out of range")
        
        new_amplitudes = {}
        factor = 1.0 / np.sqrt(2)
        
        for state, amplitude in self.amplitudes.items():
            bit_value = (state >> target) & 1
            # State with target bit = 0
            zero_state = state & ~(1 << target)
            # State with target bit = 1
            one_state = state | (1 << target)
            
            if bit_value == 0:
                new_amplitudes[zero_state] = new_amplitudes.get(zero_state, 0) + factor * amplitude
                new_amplitudes[one_state] = new_amplitudes.get(one_state, 0) + factor * amplitude
            else:
                new_amplitudes[zero_state] = new_amplitudes.get(zero_state, 0) + factor * amplitude
                new_amplitudes[one_state] = new_amplitudes.get(one_state, 0) - factor * amplitude
        
        self.amplitudes = new_amplitudes
    
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate with control and target qubits"""
        if control >= self.num_qubits or target >= self.num_qubits:
            raise ValueError(f"Qubit indices out of range")
        
        new_amplitudes = {}
        
        for state, amplitude in self.amplitudes.items():
            control_bit = (state >> control) & 1
            if control_bit == 1:
                # Flip target bit
                target_bit = (state >> target) & 1
                new_state = state ^ (1 << target)  # XOR to flip bit
                new_amplitudes[new_state] = amplitude
            else:
                # Keep state unchanged
                new_amplitudes[state] = amplitude
        
        self.amplitudes = new_amplitudes
    
    def apply_string_tension(self, tension: float):
        """Apply string-theory-inspired tension to entangle states"""
        # Simplified model: apply phase shifts based on Hamming weight
        for state in list(self.amplitudes.keys()):
            # Count set bits (Hamming weight)
            hamming_weight = bin(state).count('1')
            # Apply phase based on tension and Hamming weight
            phase_shift = np.exp(1j * tension * hamming_weight / self.num_qubits)
            self.amplitudes[state] *= phase_shift
    
    def get_entropy(self) -> float:
        """Calculate von Neumann entropy of the state"""
        # For simplicity, use classical Shannon entropy of probability distribution
        probs = [np.abs(amp)**2 for amp in self.amplitudes.values()]
        total_prob = sum(probs)
        if total_prob < 1e-9:
            return 0.0
        
        # Normalize probabilities
        probs = [p/total_prob for p in probs]
        
        # Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 1e-9:  # Avoid log(0)
                entropy -= p * np.log2(p)
        
        return entropy

# ... rest of the implementation will be written to separate files
EOL

# Create the API server file
echo -e "${GREEN}Creating API server...${NC}"
cat > api_server.py << 'EOL'
#!/usr/bin/env python3
"""
API Server for Quantum Kaleidoscope
===================================

Provides HTTP API for interfacing with the Quantum Kaleidoscope engine.
"""

import os
import json
import time
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KaleidoscopeAPI")

# Create the Flask app
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Global reference to the Kaleidoscope system
kaleidoscope_system = None
background_thread = None
thread_lock = threading.Lock()

def background_thread():
    """Send server-side events to clients."""
    count = 0
    while True:
        socketio.sleep(1)
        if kaleidoscope_system:
            count += 1
            if count % 2 == 0:  # Send updates every 2 seconds
                viz_data = kaleidoscope_system.get_visualization_data()
                socketio.emit('system_update', viz_data)

@app.route("/")
def home():
    """Serve the main HTML page"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system status"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    status = kaleidoscope_system.get_status()
    return jsonify(status)

@app.route("/api/visualization", methods=["GET"])
def get_visualization_data():
    """Get visualization data"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    viz_data = kaleidoscope_system.get_visualization_data()
    return jsonify(viz_data)

@app.route("/api/process/text", methods=["POST"])
def process_text():
    """Process text input"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing text in request"}), 400
    
    text = data["text"]
    metadata = data.get("metadata", {})
    
    result = kaleidoscope_system.process_text(text, metadata)
    return jsonify(result)

@app.route("/api/nodes/create", methods=["POST"])
def create_node():
    """Create a new node"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "Missing features in request"}), 400
    
    features = np.array(data["features"])
    position = np.array(data.get("position")) if "position" in data else None
    energy = float(data.get("energy", 0.5))
    stability = float(data.get("stability", 0.8))
    
    node_id = kaleidoscope_system.engine.create_node(
        features=features,
        position=position,
        energy=energy,
        stability=stability
    )
    
    return jsonify({"node_id": node_id})

@app.route("/api/nodes/connect", methods=["POST"])
def connect_nodes():
    """Connect two nodes"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    data = request.json
    if not data or "node1_id" not in data or "node2_id" not in data:
        return jsonify({"error": "Missing node IDs in request"}), 400
    
    node1_id = data["node1_id"]
    node2_id = data["node2_id"]
    strength = float(data["strength"]) if "strength" in data else None
    
    result = kaleidoscope_system.engine.connect_nodes(node1_id, node2_id, strength)
    return jsonify({"success": result})

@app.route("/api/auto-generation/start", methods=["POST"])
def start_auto_generation():
    """Start auto-generation"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    data = request.json or {}
    interval = float(data.get("interval", 5.0))
    
    kaleidoscope_system.start_auto_generation(interval=interval)
    return jsonify({"success": True})

@app.route("/api/auto-generation/stop", methods=["POST"])
def stop_auto_generation():
    """Stop auto-generation"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    kaleidoscope_system.stop_auto_generation()
    return jsonify({"success": True})

@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """Run simulation steps"""
    if not kaleidoscope_system:
        return jsonify({"error": "System not initialized"}), 500
    
    data = request.json or {}
    steps = int(data.get("steps", 1))
    
    for _ in range(steps):
        kaleidoscope_system.engine.run_simulation_step()
    
    return jsonify({"success": True})

@socketio.on('connect')
def handle_connect():
    global background_thread
    with thread_lock:
        if background_thread is None:
            background_thread = socketio.start_background_task(background_thread)
    
    # Send initial data
    if kaleidoscope_system:
        viz_data = kaleidoscope_system.get_visualization_data()
        emit('system_update', viz_data)

def start_server(system, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    global kaleidoscope_system
    kaleidoscope_system = system
    
    # Start Flask-SocketIO app
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Kaleidoscope API Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    try:
        # Try to import the system
        from quantum_kaleidoscope import QuantumKaleidoscope
        
        print("Initializing Quantum Kaleidoscope system...")
        kaleidoscope_system = QuantumKaleidoscope(dimension=128)
        
        print(f"Starting API server on port {args.port}")
        start_server(kaleidoscope_system, port=args.port)
        
    except ImportError:
        print("Error: Unable to import QuantumKaleidoscope. Please make sure the core system is available.")
        sys.exit(1)
EOL

# Create a simple index.html
echo -e "${GREEN}Creating frontend interface...${NC}"
cat > static/index.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Kaleidoscope</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #0a0a1f;
            color: #e8e8e8;
        }
        #app {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        header {
            background: linear-gradient(90deg, #4A6BFF, #BD00FF);
            padding: 15px;
            color: white;
            text-align: center;
        }
        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .side-panel {
            width: 300px;
            background-color: rgba(20, 25, 60, 0.8);
            padding: 15px;
            overflow-y: auto;
        }
        .visualization {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #040414;
        }
        .panel {
            background-color: rgba(30, 35, 70, 0.7);
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
        }
        h2 {
            margin-top: 0;
            color: #00f3ff;
            font-size: 18px;
        }
        button {
            background-color: #4A6BFF;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #3A5BEF;
        }
        textarea {
            width: 100%;
            height: 100px;
            background-color: rgba(0, 0, 0, 0.3);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 8px;
            resize: vertical;
            font-family: inherit;
            margin-bottom: 10px;
        }
        #canvas {
            width: 100%;
            height: 100%;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .metric {
            background-color: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .metric-label {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.7);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4A6BFF;
        }
        #status {
            position: absolute;
            bottom: 10px;
            right: 10px;
            padding: 5px 10px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 4px;
            font-size: 12px;
        }
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 999;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #4A6BFF;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div id="app">
        <header>
            <h1>Quantum Kaleidoscope</h1>
        </header>
        <div class="main-container">
            <div class="side-panel">
                <div class="panel">
                    <h2>System Metrics</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Nodes</div>
                            <div id="node-count" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Patterns</div>
                            <div id="pattern-count" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Insights</div>
                            <div id="insight-count" class="metric-value">0</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Perspectives</div>
                            <div id="perspective-count" class="metric-value">0</div>
                        </div>
                    </div>
                </div>
                <div class="panel">
                    <h2>Process Text</h2>
                    <textarea id="input-text" placeholder="Enter text to process..."></textarea>
                    <button id="process-btn">Process</button>
                </div>
                <div class="panel">
                    <h2>Controls</h2>
                    <button id="auto-gen-btn">Start Auto-Generation</button>
                    <button id="run-step-btn">Run Simulation Step</button>
                </div>
            </div>
            <div class="visualization">
                <canvas id="canvas"></canvas>
                <div id="status">Ready</div>
            </div>
        </div>
    </div>
    
    <div class="loading" id="loading" style="display: none;">
        <div class="spinner"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Initialize the application
        document.addEventListener('DOMContentLoaded', () => {
            const socket = io();
            let isAutoGenActive = false;
            
            // DOM Elements
            const nodeCountEl = document.getElementById('node-count');
            const patternCountEl = document.getElementById('pattern-count');
            const insightCountEl = document.getElementById('insight-count');
            const perspectiveCountEl = document.getElementById('perspective-count');
            const inputTextEl = document.getElementById('input-text');
            const processBtnEl = document.getElementById('process-btn');
            const autoGenBtnEl = document.getElementById('auto-gen-btn');
            const runStepBtnEl = document.getElementById('run-step-btn');
            const statusEl = document.getElementById('status');
            const loadingEl = document.getElementById('loading');
            
            // Three.js Setup
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;
            
            const canvas = document.getElementById('canvas');
            const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Objects to track
            const nodes = {};
            const connections = [];
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                
                // Add any animations here
                
                renderer.render(scene, camera);
            }
            animate();
            
            // Handle window resize
            window.addEventListener('resize', () => {
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            });
            
            // Socket.io event handlers
            socket.on('connect', () => {
                statusEl.textContent = 'Connected';
                statusEl.style.color = '#4CAF50';
            });
            
            socket.on('disconnect', () => {
                statusEl.textContent = 'Disconnected';
                statusEl.style.color = '#F44336';
            });
            
            socket.on('system_update', (data) => {
                updateVisualization(data);
                updateMetrics(data.metrics);
            });
            
            // API Calls
            async function processText() {
                const text = inputTextEl.value.trim();
                if (!text) return;
                
                loadingEl.style.display = 'flex';
                
                try {
                    const response = await fetch('/api/process/text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                    
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const result = await response.json();
                    console.log('Processing result:', result);
                    
                    // Clear input
                    inputTextEl.value = '';
                    
                } catch (error) {
                    console.error('Error processing text:', error);
                    statusEl.textContent = 'Error processing text';
                    statusEl.style.color = '#F44336';
                } finally {
                    loadingEl.style.display = 'none';
                }
            }
            
            async function toggleAutoGeneration() {
                loadingEl.style.display = 'flex';
                
                try {
                    if (isAutoGenActive) {
                        await fetch('/api/auto-generation/stop', { method: 'POST' });
                        autoGenBtnEl.textContent = 'Start Auto-Generation';
                        isAutoGenActive = false;
                    } else {
                        await fetch('/api/auto-generation/start', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ interval: 3.0 }),
                        });
                        autoGenBtnEl.textContent = 'Stop Auto-Generation';
                        isAutoGenActive = true;
                    }
                } catch (error) {
                    console.error('Error toggling auto-generation:', error);
                } finally {
                    loadingEl.style.display = 'none';
                }
            }
            
            async function runSimulationStep() {
                loadingEl.style.display = 'flex';
                
                try {
                    await fetch('/api/simulate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ steps: 1 }),
                    });
                } catch (error) {
                    console.error('Error running simulation step:', error);
                } finally {
                    loadingEl.style.display = 'none';
                }
            }
            
            // Visualization update
            function updateVisualization(data) {
                // Clear previous objects
                for (const nodeId in nodes) {
                    scene.remove(nodes[nodeId]);
                }
                
                for (const connection of connections) {
                    scene.remove(connection);
                }
                
                connections.length = 0;
                
                // Add nodes
                if (data.nodes) {
                    for (const node of data.nodes) {
                        const geometry = new THREE.SphereGeometry(0.1 * (node.energy + 0.5), 16, 16);
                        const material = new THREE.MeshPhongMaterial({
                            color: new THREE.Color(0.2, 0.5 + node.stability * 0.5, 1.0),
                            emissive: new THREE.Color(0.1, 0.2, 0.5),
                            shininess: 100
                        });
                        
                        const mesh = new THREE.Mesh(geometry, material);
                        mesh.position.set(...node.position);
                        scene.add(mesh);
                        
                        nodes[node.id] = mesh;
                    }
                }
                
                // Add connections
                if (data.connections) {
                    for (const conn of data.connections) {
                        if (nodes[conn.source] && nodes[conn.target]) {
                            const sourcePos = nodes[conn.source].position;
                            const targetPos = nodes[conn.target].position;
                            
                            const points = [
                                new THREE.Vector3(sourcePos.x, sourcePos.y, sourcePos.z),
                                new THREE.Vector3(targetPos.x, targetPos.y, targetPos.z)
                            ];
                            
                            const geometry = new THREE.BufferGeometry().setFromPoints(points);
                            const material = new THREE.LineBasicMaterial({
                                color: 0x4A6BFF,
                                opacity: conn.strength,
                                transparent: true
                            });
                            
                            const line = new THREE.Line(geometry, material);
                            scene.add(line);
                            connections.push(line);
                        }
                    }
                }
            }
            
            // Update metrics display
            function updateMetrics(metrics) {
                if (metrics) {
                    nodeCountEl.textContent = metrics.node_count || 0;
                    patternCountEl.textContent = metrics.pattern_count || 0;
                    insightCountEl.textContent = metrics.insight_count || 0;
                    perspectiveCountEl.textContent = metrics.perspective_count || 0;
                }
            }
            
            // Event listeners
            processBtnEl.addEventListener('click', processText);
            autoGenBtnEl.addEventListener('click', toggleAutoGeneration);
            runStepBtnEl.addEventListener('click', runSimulationStep);
            
            // Initial fetch
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    console.log('System status:', data);
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        });
    </script>
</body>
</html>
EOL

echo -e "${GREEN}Setup completed successfully!${NC}"
echo "You can run the system with: python3 api_server.py"
echo "Then navigate to http://localhost:${PORT} in your browser"

# Run the server if requested
if [[ "$1" == "--run" ]]; then
    cd "$INSTALL_DIR"
    echo -e "${GREEN}Starting server on port ${PORT}...${NC}"
    python3 api_server.py --port ${PORT}
fi