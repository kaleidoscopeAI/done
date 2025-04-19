#!/usr/bin/env python3
"""
Quantum Kaleidoscope: Integrated Cognitive Intelligence System
=============================================================

Core engine with quantum-inspired processing, advanced pattern recognition,
and multidimensional data analysis capabilities.

(Ensure all necessary imports like numpy, logging, etc. are present)
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

# Configure logging (can be done here or rely on run_system.py's config)
# If configured here, use getLogger to avoid conflicts
logger = logging.getLogger("QuantumKaleidoscope")
if not logger.hasHandlers(): # Avoid adding handlers multiple times if imported
     logging.basicConfig(
         level=logging.INFO,
         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
         datefmt="%Y-%m-%d %H:%M:%S"
     )


# ===================================
# Core Data Structures (ResonanceMode, SuperNodeDNA, etc.)
# ===================================
class ResonanceMode(Enum):
    LINEAR = auto(); NONLINEAR = auto(); QUANTUM = auto(); CHAOTIC = auto(); TOPOLOGICAL = auto(); HYBRID = auto()

@dataclass
class SuperNodeDNA:
    encoding: np.ndarray
    linear_weight: float = 0.2; nonlinear_weight: float = 0.3; quantum_weight: float = 0.1
    chaotic_weight: float = 0.1; topological_weight: float = 0.3
    mutation_rate: float = 0.005; crossover_points: int = 3
    connectivity_pattern: np.ndarray = field(default_factory=lambda: np.random.rand(64))
    activation_thresholds: np.ndarray = field(default_factory=lambda: np.random.rand(16) * 0.5 + 0.3)

    def evolve(self) -> 'SuperNodeDNA':
        evolved = SuperNodeDNA(encoding=self.encoding.copy(), linear_weight=self.linear_weight, nonlinear_weight=self.nonlinear_weight,
                              quantum_weight=self.quantum_weight, chaotic_weight=self.chaotic_weight, topological_weight=self.topological_weight,
                              mutation_rate=self.mutation_rate, crossover_points=self.crossover_points,
                              connectivity_pattern=self.connectivity_pattern.copy(), activation_thresholds=self.activation_thresholds.copy())
        mutation_mask = np.random.rand(len(evolved.encoding)) < self.mutation_rate
        mutation_strength = np.random.randn(len(evolved.encoding)) * 0.1
        evolved.encoding[mutation_mask] += mutation_strength[mutation_mask]
        norm = np.linalg.norm(evolved.encoding); evolved.encoding /= norm if norm > 1e-10 else 1.0
        weight_mutation = np.random.randn(5) * 0.02
        evolved.linear_weight = max(0.1, min(0.5, evolved.linear_weight + weight_mutation[0]))
        evolved.nonlinear_weight = max(0.1, min(0.5, evolved.nonlinear_weight + weight_mutation[1]))
        evolved.quantum_weight = max(0.05, min(0.3, evolved.quantum_weight + weight_mutation[2]))
        evolved.chaotic_weight = max(0.05, min(0.3, evolved.chaotic_weight + weight_mutation[3]))
        evolved.topological_weight = max(0.1, min(0.5, evolved.topological_weight + weight_mutation[4]))
        total = (evolved.linear_weight + evolved.nonlinear_weight + evolved.quantum_weight + evolved.chaotic_weight + evolved.topological_weight)
        if total > 0:
             evolved.linear_weight /= total; evolved.nonlinear_weight /= total; evolved.quantum_weight /= total
             evolved.chaotic_weight /= total; evolved.topological_weight /= total
        conn_mutation_mask = np.random.rand(len(evolved.connectivity_pattern)) < self.mutation_rate * 2
        conn_mutation = np.random.randn(len(evolved.connectivity_pattern)) * 0.1
        evolved.connectivity_pattern[conn_mutation_mask] += conn_mutation[conn_mutation_mask]
        evolved.connectivity_pattern = np.clip(evolved.connectivity_pattern, 0, 1)
        threshold_mutation_mask = np.random.rand(len(evolved.activation_thresholds)) < self.mutation_rate * 2
        threshold_mutation = np.random.randn(len(evolved.activation_thresholds)) * 0.05
        evolved.activation_thresholds[threshold_mutation_mask] += threshold_mutation[threshold_mutation_mask]
        evolved.activation_thresholds = np.clip(evolved.activation_thresholds, 0.1, 0.9)
        return evolved

    @staticmethod
    def generate(dimension: int = 512) -> 'SuperNodeDNA':
        encoding = np.random.randn(dimension); encoding /= np.linalg.norm(encoding) if np.linalg.norm(encoding) > 1e-10 else 1.0
        return SuperNodeDNA(encoding=encoding, linear_weight=0.2, nonlinear_weight=0.3, quantum_weight=0.1, chaotic_weight=0.1, topological_weight=0.3)

@dataclass
class SuperNodeState:
    current: np.ndarray
    memory: deque = field(default_factory=lambda: deque(maxlen=256))
    attractors: List[np.ndarray] = field(default_factory=list)
    energy: float = 1.0; stability: float = 0.0; coherence: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    def update(self, new_state: np.ndarray) -> None:
        if self.current is not None and len(self.current) == len(new_state):
            self.memory.append(self.current.copy())
            norm_curr = np.linalg.norm(self.current); norm_new = np.linalg.norm(new_state)
            if norm_curr > 1e-10 and norm_new > 1e-10:
                stability = np.dot(self.current, new_state) / (norm_curr * norm_new)
                self.stability = 0.8 * self.stability + 0.2 * stability
        self.current = new_state.copy()
        if len(self.memory) > 5:
            recent = list(self.memory)[-5:]; similarities = []
            for i in range(len(recent)):
                for j in range(i+1, len(recent)):
                    norm1 = np.linalg.norm(recent[i]); norm2 = np.linalg.norm(recent[j])
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        sim = np.dot(recent[i], recent[j]) / (norm1 * norm2)
                        similarities.append(sim)
            self.coherence = np.mean(similarities) if similarities else 0.0
        self.energy = max(0.0, self.energy * 0.998) # Prevent negative energy
        self.history.append({"time": time.time(), "stability": self.stability, "coherence": self.coherence, "energy": self.energy})
        self.last_update = time.time()

    def add_attractor(self, attractor: np.ndarray) -> None:
        norm_attr = np.linalg.norm(attractor)
        if norm_attr < 1e-10: return # Don't add zero vector
        attractor = attractor / norm_attr # Normalize before adding
        for existing in self.attractors:
            norm_exist = np.linalg.norm(existing)
            if norm_exist > 1e-10:
                similarity = np.dot(attractor, existing) / norm_exist # Existing is already normalized
                if similarity > 0.95: return # Higher threshold for adding
        self.attractors.append(attractor.copy())
        if len(self.attractors) > 10: self.attractors.pop(0)

@dataclass
class Pattern:
    id: str; type: str; vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5; timestamp: float = field(default_factory=time.time)

    def similarity(self, other: 'Pattern') -> float:
        vec1, vec2 = self.vector, other.vector
        if vec1.shape != vec2.shape:
            min_dim = min(len(vec1), len(vec2)); vec1 = vec1[:min_dim]; vec2 = vec2[:min_dim]
        norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10: return 0.0
        return float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)) # Clip cosine similarity

@dataclass
class Insight:
    id: str; type: str; patterns: List[str]; vector: np.ndarray; description: str
    confidence: float = 0.5; importance: float = 0.5; novelty: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Perspective:
    id: str; insight_ids: List[str]; vector: np.ndarray; strength: float; coherence: float
    novelty: float; impact: float; description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        # Exclude vector from default dict representation for brevity
        return {k: v for k, v in asdict(self).items() if k != 'vector'}

@dataclass
class QuantumState:
    num_qubits: int
    amplitudes: Dict[int, complex] = field(default_factory=dict)

    def __post_init__(self):
        if not self.amplitudes: self.amplitudes = {0: complex(1.0, 0.0)}

    def apply_hadamard(self, target: int):
        if target >= self.num_qubits: return
        new_amplitudes = {}; norm_factor = 1.0 / np.sqrt(2.0)
        touched_indices = set(self.amplitudes.keys())
        processed = set()
        for idx in touched_indices:
             if idx in processed: continue
             paired_idx = idx ^ (1 << target)
             amp_idx = self.amplitudes.get(idx, 0)
             amp_paired = self.amplitudes.get(paired_idx, 0)

             # Calculate new amplitudes for idx and paired_idx based on HAD formula
             new_amp_idx = (amp_idx + amp_paired) * norm_factor if ((idx >> target) & 1) == 0 else (amp_idx - amp_paired) * norm_factor
             new_amp_paired = (amp_idx + amp_paired) * norm_factor if ((paired_idx >> target) & 1) == 0 else (amp_idx - amp_paired) * norm_factor

             if abs(new_amp_idx) > 1e-10: new_amplitudes[idx] = new_amp_idx
             if abs(new_amp_paired) > 1e-10: new_amplitudes[paired_idx] = new_amp_paired
             processed.add(idx); processed.add(paired_idx)
        self.amplitudes = new_amplitudes

    def apply_cnot(self, control: int, target: int):
        if control >= self.num_qubits or target >= self.num_qubits: return
        new_amplitudes = {}
        for idx, amp in self.amplitudes.items():
            if (idx >> control) & 1: flipped = idx ^ (1 << target); new_amplitudes[flipped] = amp
            else: new_amplitudes[idx] = amp
        self.amplitudes = new_amplitudes

    def apply_string_tension(self, tension: float):
        new_amplitudes = {}; norm_factor_sq = 0.0
        max_tension_effect = 0.5 # Limit tension effect to prevent blowup
        effective_tension = np.clip(tension, -max_tension_effect, max_tension_effect)
        for idx, amp in self.amplitudes.items():
            hamming = bin(idx).count('1')
            scale = 1.0 + (hamming / self.num_qubits - 0.5) * effective_tension # Use clipped tension
            new_amp = amp * scale
            new_amplitudes[idx] = new_amp
            norm_factor_sq += abs(new_amp)**2
        norm_factor = math.sqrt(norm_factor_sq) if norm_factor_sq > 0 else 1.0
        self.amplitudes = {k: v / norm_factor for k, v in new_amplitudes.items() if abs(v) > 1e-10} # Filter small amps

    def get_entropy(self) -> float:
        entropy = 0.0
        for amp in self.amplitudes.values():
            prob = abs(amp)**2
            if prob > 1e-10: entropy -= prob * math.log2(prob) # Use log base 2 for bits
        return entropy / self.num_qubits if self.num_qubits > 0 else 0.0 # Normalize entropy

# ===================================
# Quantum String Cube
# ===================================
class StringCube:
    def __init__(self, dimension: int = 3, resolution: int = 10):
        self.dimension = dimension; self.resolution = resolution
        shape = [resolution] * dimension
        self.grid = np.zeros(shape, dtype=np.float32)
        self.tension_field = np.zeros(shape, dtype=np.float32)
        self.tensor_fields = np.zeros(shape + [dimension, dimension], dtype=np.float32)
        self.quantum_phase_grid = np.ones(shape, dtype=np.complex128) # Initialize phase to 1 (angle 0)
        self.nodes_map = {}
        self.tension_strength = 0.5; self.elasticity = 0.3; self.damping = 0.95

    def add_node(self, node: 'ConsciousNode') -> Tuple:
        grid_pos = self._continuous_to_grid(node.position)
        if grid_pos not in self.nodes_map: self.nodes_map[grid_pos] = []
        if node.id not in self.nodes_map[grid_pos]: self.nodes_map[grid_pos].append(node.id) # Avoid duplicates
        self.grid[grid_pos] = max(self.grid[grid_pos], node.energy * 0.1) # Use max or avg? Max emphasizes peaks.
        phase = node.stability * 2 * np.pi; self.quantum_phase_grid[grid_pos] *= np.exp(1j * phase) # Multiply phases
        self._normalize_phase(grid_pos)
        return grid_pos

    def _continuous_to_grid(self, position: np.ndarray) -> Tuple:
        pos = np.clip(position[:self.dimension], -1.0, 1.0) # Ensure position is within bounds
        grid_coords = tuple(int(np.clip((p + 1) / 2 * (self.resolution - 1), 0, self.resolution - 1)) for p in pos)
        # Pad if needed (should match self.dimension)
        while len(grid_coords) < self.dimension: grid_coords += (0,)
        return grid_coords

    def update_tension(self, nodes: Dict[str, 'ConsciousNode']):
        self.tension_field *= self.damping # Apply damping first
        self.tensor_fields *= self.damping # Dampen tensor fields too
        temp_tension_updates = np.zeros_like(self.tension_field)
        temp_tensor_updates = np.zeros_like(self.tensor_fields)

        processed_connections = set() # Avoid processing A->B and B->A separately

        for node_id, node in nodes.items():
            grid_pos = self._continuous_to_grid(node.position)
            self.grid[grid_pos] = max(self.grid[grid_pos]*0.8, node.energy * 0.1) # Add decay to grid energy

            for conn_id, strength in node.connections.items():
                conn_key = tuple(sorted((node_id, conn_id)))
                if conn_key in processed_connections or conn_id not in nodes: continue
                processed_connections.add(conn_key)

                conn_node = nodes[conn_id]; conn_pos = self._continuous_to_grid(conn_node.position)
                if grid_pos == conn_pos: continue # Skip self-connections in cube

                tension_vector_grid = np.array(conn_pos) - np.array(grid_pos)
                dist_grid = np.linalg.norm(tension_vector_grid)
                if dist_grid < 1: continue # Skip adjacent grid points? Or handle differently?

                # Update tension field along the path (using temp grid)
                path_points = self._generate_path_points(grid_pos, conn_pos, steps=max(int(dist_grid), 2))
                for idx, point in enumerate(path_points):
                     if all(0 <= p < self.resolution for p in point):
                         interp = idx / (len(path_points) -1) if len(path_points) > 1 else 0.5
                         tension_contrib = strength * self.tension_strength * (1 - abs(interp - 0.5)*2) # Max tension mid-path
                         temp_tension_updates[point] += tension_contrib
                         # Update tensor fields along path (using temp grid)
                         tension_tensor = self._calculate_tension_tensor(grid_pos, conn_pos, point, tension_contrib)
                         temp_tensor_updates[point] += tension_tensor

        self.tension_field += temp_tension_updates
        self.tensor_fields += temp_tensor_updates
        # Normalize fields after updates
        max_tension = np.max(np.abs(self.tension_field)); self.tension_field /= max_tension if max_tension > 1e-6 else 1.0
        max_tensor_norm = np.max(np.linalg.norm(self.tensor_fields, axis=(-2,-1))); self.tensor_fields /= max_tensor_norm if max_tensor_norm > 1e-6 else 1.0

    def _generate_path_points(self, start: Tuple, end: Tuple, steps: int) -> List[Tuple]:
        points = set() # Use set to avoid duplicate points
        for step in range(steps + 1):
            t = step / steps
            point = tuple(int(round(s + t * (e - s))) for s, e in zip(start, end))
            points.add(point)
        return list(points) # Return unique points

    def _calculate_tension_tensor(self, pos1: Tuple, pos2: Tuple, point: Tuple, tension: float) -> np.ndarray:
        v1 = np.array(pos1) - np.array(point); v2 = np.array(pos2) - np.array(point)
        norm1 = np.linalg.norm(v1); norm2 = np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            v1 = v1 / norm1; v2 = v2 / norm2
            # Use outer product for directionality
            tensor = tension * np.outer(v1[:self.dimension], v2[:self.dimension])
            # Symmetrize
            tensor = (tensor + tensor.T) / 2
            return tensor
        return np.zeros((self.dimension, self.dimension))

    def apply_tension_to_nodes(self, nodes: Dict[str, 'ConsciousNode']):
        for node in nodes.values():
            grid_pos = self._continuous_to_grid(node.position)
            if all(0 <= p < self.resolution for p in grid_pos):
                tension = float(self.tension_field[grid_pos])
                node.quantum_state.apply_string_tension(tension) # Apply to quantum state
                energy_change = tension * self.elasticity * node.stability * 0.1 # Scaled change
                node.energy = np.clip(node.energy + energy_change, 0.01, 1.0)
                node.stability = np.clip(node.stability * (1.0 - 0.01 * abs(tension)), 0.1, 0.99)
                # Position change based on tensor field (stress direction)
                position_change = self._calculate_position_change(grid_pos, node.stability)
                node.position = np.clip(node.position + position_change * self.elasticity * 0.05, -1.0, 1.0) # Smaller position change factor

    def _calculate_position_change(self, grid_pos: Tuple, stability: float) -> np.ndarray:
        if not all(0 <= p < self.resolution for p in grid_pos): return np.zeros(self.dimension)
        tensor = self.tensor_fields[grid_pos]
        try:
             eigenvalues, eigenvectors = np.linalg.eigh(tensor) # Use eigh for symmetric
             if len(eigenvalues) == 0: return np.zeros(self.dimension)
             max_idx = np.argmax(np.abs(eigenvalues)) # Direction of max stress/strain
             direction = eigenvectors[:, max_idx]
             # Scale by eigenvalue magnitude and inverse stability (less stable nodes move more)
             scale = eigenvalues[max_idx] * (1.0 - stability) * 0.1 # Adjusted scaling
             # Ensure direction matches dimension
             if len(direction) < self.dimension:
                  padded_dir = np.zeros(self.dimension); padded_dir[:len(direction)] = direction; direction = padded_dir
             elif len(direction) > self.dimension:
                  direction = direction[:self.dimension]
             return direction * scale
        except np.linalg.LinAlgError:
             logger.debug(f"Eigendecomposition failed for tensor at {grid_pos}. No position change.")
             return np.zeros(self.dimension)

    def calculate_scalar_tension_field(self) -> np.ndarray:
        # Use Frobenius norm of the tensor field for scalar tension
        tension = np.linalg.norm(self.tensor_fields, axis=(-2,-1))
        max_tension = np.max(tension); tension /= max_tension if max_tension > 1e-6 else 1.0
        return tension

    def evolve_quantum_phase_grid(self):
        tension = self.calculate_scalar_tension_field() # Use tensor norm based tension
        # Phase evolves faster in high tension, slower/damped in low tension
        phase_change = (tension - 0.5) * 0.1 * np.pi # Center change around tension=0.5
        self.quantum_phase_grid *= np.exp(1j * phase_change)
        # Normalize phases periodically to prevent drift? Or just handle magnitude?
        # Let's ensure magnitude is 1
        np.divide(self.quantum_phase_grid, np.abs(self.quantum_phase_grid), out=self.quantum_phase_grid, where=np.abs(self.quantum_phase_grid)>1e-10)

    def _normalize_phase(self, grid_pos):
         mag = np.abs(self.quantum_phase_grid[grid_pos])
         if mag > 1e-10: self.quantum_phase_grid[grid_pos] /= mag

# ===================================
# Conscious Node
# ===================================
@dataclass
class ConsciousNode:
    id: str; position: np.ndarray; energy: float; stability: float; features: np.ndarray
    connections: Dict[str, float] = field(default_factory=dict)
    memory: List[np.ndarray] = field(default_factory=list) # Could use deque
    data: Dict[str, Any] = field(default_factory=dict)
    quantum_state: Optional[QuantumState] = None

    def __post_init__(self):
        if self.quantum_state is None:
            num_qubits = 8 # Default qubits
            self.quantum_state = QuantumState(num_qubits)
            for i in range(num_qubits): self.quantum_state.apply_hadamard(i) # Initial superposition

    def update_energy(self, decay: float):
        self.energy = max(0.0, self.energy * decay) # Decay
        entropy = self.quantum_state.get_entropy() # Get normalized entropy
        energy_fluctuation = (np.random.random() - 0.5) * 0.01 * entropy * self.energy # Scale fluctuation by energy & entropy
        self.energy = np.clip(self.energy + energy_fluctuation, 0.01, 1.0) # Apply fluctuation and clip
        return self.energy

    def calculate_affinity(self, other_node: 'ConsciousNode') -> float:
        # Ensure features are normalized for dot product similarity
        norm_self = np.linalg.norm(self.features); norm_other = np.linalg.norm(other_node.features)
        if norm_self < 1e-10 or norm_other < 1e-10: feature_similarity = 0.0
        else: feature_similarity = np.dot(self.features / norm_self, other_node.features / norm_other)

        # Ensure positions have same dimension for distance calculation
        dim = min(len(self.position), len(other_node.position))
        position_distance = np.linalg.norm(self.position[:dim] - other_node.position[:dim])
        position_factor = np.exp(-position_distance) # Exponential decay with distance

        energy_diff = abs(self.energy - other_node.energy)
        energy_factor = max(0.0, 1.0 - energy_diff) # Closer energy -> higher factor

        # Weighted combination
        affinity = 0.5 * feature_similarity + 0.3 * position_factor + 0.2 * energy_factor
        return float(np.clip(affinity, 0.0, 1.0))

# ===================================
# SuperNode Core
# ===================================
class SuperNodeCore:
    def __init__(self, dimension: int = 512, resonance_mode: ResonanceMode = ResonanceMode.HYBRID, auto_tune: bool = True):
        self.dimension = dimension; self.resonance_mode = resonance_mode; self.auto_tune = auto_tune
        self.dna = SuperNodeDNA.generate(dimension); self.state = SuperNodeState(current=np.zeros(dimension))
        self.knowledge_base = deque(maxlen=200) # Use deque for efficient addition/removal
        self.resonance_matrix = np.eye(dimension)
        self.evolution_active = False; self.evolution_thread = None; self.evolution_interval = 1.0
        self.evolution_stop_event = threading.Event()
        self.metrics = {"resonance_stability": 0.0, "knowledge_coherence": 0.0, "evolution_rate": 0.001,
                        "processing_count": 0, "significant_patterns": 0, "pattern_count": 0, "insight_count": 0, "perspective_count": 0}
        # Use logger specific to this instance
        self.logger = logging.getLogger(f"SuperNodeCore_{id(self)}")
        self.lock = threading.Lock() # Lock for thread safety if needed for DNA/state updates

    def start(self) -> None:
        if self.evolution_active: self.logger.warning("Evolution already active"); return
        with self.lock: self.state.current = self.dna.encoding.copy() # Initialize state safely
        self.evolution_stop_event.clear(); self.evolution_active = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True, name=f"SNCoreEvo_{id(self)}")
        self.evolution_thread.start()
        self.logger.info("Started SuperNodeCore evolution")

    def stop(self) -> None:
        if not self.evolution_active: self.logger.warning("Evolution not active"); return
        self.evolution_stop_event.set()
        if self.evolution_thread: self.evolution_thread.join(timeout=5.0)
        self.evolution_active = False
        self.logger.info("Stopped SuperNodeCore evolution")

    def _ensure_dimension(self, vector: np.ndarray) -> np.ndarray:
         """Pads or truncates vector to match self.dimension."""
         current_dim = len(vector)
         if current_dim == self.dimension: return vector
         elif current_dim > self.dimension: return vector[:self.dimension]
         else: # current_dim < self.dimension
             padded = np.zeros(self.dimension, dtype=vector.dtype)
             padded[:current_dim] = vector
             return padded

    def process_input(self, data: np.ndarray) -> np.ndarray:
        data = self._ensure_dimension(data.flatten()) # Ensure correct shape and dimension
        norm = np.linalg.norm(data); data = data / norm if norm > 1e-10 else data

        # Choose processing function based on mode
        mode_map = {
            ResonanceMode.LINEAR: self._apply_linear_resonance,
            ResonanceMode.NONLINEAR: self._apply_nonlinear_resonance,
            ResonanceMode.QUANTUM: self._apply_quantum_resonance,
            ResonanceMode.CHAOTIC: self._apply_chaotic_resonance,
            ResonanceMode.TOPOLOGICAL: self._apply_topological_resonance,
            ResonanceMode.HYBRID: self._apply_hybrid_resonance,
        }
        processing_func = mode_map.get(self.resonance_mode, self._apply_hybrid_resonance)
        processed = processing_func(data)

        with self.lock: # Protect state update
            self.state.update(processed)
            self.metrics["processing_count"] += 1
        return processed

    def absorb_knowledge(self, knowledge_vector: np.ndarray) -> None:
        knowledge_vector = self._ensure_dimension(knowledge_vector.flatten())
        norm = np.linalg.norm(knowledge_vector); knowledge_vector = knowledge_vector / norm if norm > 1e-10 else knowledge_vector

        with self.lock: # Protect DNA access and knowledge base/state updates
            dna_influence = 0.1 * np.dot(knowledge_vector, self.dna.encoding) * self.dna.encoding
            influenced_vector = knowledge_vector + dna_influence
            norm_inf = np.linalg.norm(influenced_vector); influenced_vector /= norm_inf if norm_inf > 1e-10 else 1.0

            self.knowledge_base.append(influenced_vector) # deque handles maxlen automatically
            self._update_resonance_matrix() # Should this be inside the lock? Depends if thread accesses it.

            knowledge_influence_state = 0.05 * influenced_vector
            new_state_vec = self.state.current + knowledge_influence_state
            norm_new_state = np.linalg.norm(new_state_vec); new_state_vec /= norm_new_state if norm_new_state > 1e-10 else 1.0
            self.state.update(new_state_vec)

    def _evolution_loop(self) -> None:
        while not self.evolution_stop_event.is_set():
            try:
                evolve_dna_flag = False
                update_matrix_flag = False
                apply_stability_flag = False
                with self.lock: # Read metrics safely
                    should_evolve_dna = np.random.random() < self.metrics.get("evolution_rate", 0.001)
                    should_update_matrix = np.random.random() < 0.2
                    should_apply_stability = len(self.state.memory) > 5

                if should_evolve_dna:
                    # Evolve DNA outside the lock to avoid holding it too long
                    evolved_dna = self.dna.evolve() # Assuming dna.evolve() is thread-safe or reads self.dna once
                    # Re-acquire lock to check stability and potentially update DNA
                    with self.lock:
                        stability_trend = self._get_stability_trend() # Assumes history read is ok here
                        # More likely to evolve if stability is low or decreasing
                        evolve_prob = 0.1 + 0.4 * max(0, -stability_trend * 10) + 0.3 * (1 - self.state.stability)
                        if np.random.random() < evolve_prob:
                            self.dna = evolved_dna
                            evolve_dna_flag = True # Mark that DNA was updated

                if should_update_matrix:
                    with self.lock: # Update resonance matrix under lock
                        self._update_resonance_matrix()
                        update_matrix_flag = True

                if should_apply_stability:
                    with self.lock: # Apply stability influence under lock
                        self._apply_stability_influence()
                        apply_stability_flag = True

                # Log actions taken (optional)
                # if evolve_dna_flag or update_matrix_flag or apply_stability_flag:
                #     self.logger.debug(f"EvoLoop: DNA={evolve_dna_flag}, Matrix={update_matrix_flag}, Stability={apply_stability_flag}")

                # Sleep (use wait for better responsiveness to stop event)
                self.evolution_stop_event.wait(self.evolution_interval)

            except Exception as e:
                self.logger.error(f"Error in evolution loop: {e}", exc_info=True)
                self.evolution_stop_event.wait(5.0) # Sleep longer on error

    def _update_resonance_matrix(self) -> None:
        # This method should be called *with the lock held* if knowledge_base can be modified concurrently
        if not self.knowledge_base: self.resonance_matrix = np.eye(self.dimension); return
        base_matrix = 0.8 * np.eye(self.dimension)
        # Use recent knowledge vectors from deque
        recent_knowledge = list(self.knowledge_base)[-20:]
        for vector in recent_knowledge:
            outer = np.outer(vector, vector); base_matrix += 0.01 * outer
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(base_matrix)
            # Clip eigenvalues to ensure stability |lambda| <= 1
            eigenvalues = np.clip(eigenvalues, -1.0, 1.0)
            # Reconstruct matrix (ensure real if original was symmetric)
            self.resonance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            self.resonance_matrix = np.real_if_close(self.resonance_matrix)
        except np.linalg.LinAlgError:
            self.logger.warning("Eigendecomposition failed in _update_resonance_matrix. Using normalization fallback.")
            spectral_radius = np.max(np.abs(np.linalg.eigvalsh(base_matrix))) # Use eigvalsh for Hermitian/symmetric
            self.resonance_matrix = base_matrix / spectral_radius if spectral_radius > 1.0 else base_matrix
        # Update metrics
        if len(recent_knowledge) >= 2:
            sims = [np.abs(np.dot(v1, v2)) for i, v1 in enumerate(recent_knowledge) for v2 in recent_knowledge[i+1:]]
            self.metrics["knowledge_coherence"] = np.mean(sims) if sims else 0.0
        else: self.metrics["knowledge_coherence"] = 0.0
        self.metrics["resonance_stability"] = np.mean(np.abs(np.diag(self.resonance_matrix)))


    def _get_stability_trend(self) -> float:
        # Assumes called with lock held if accessing self.state.history
        history = list(self.state.history) # Get copy under lock if needed
        if len(history) < 10: return 0.0
        stabilities = [h["stability"] for h in history[-10:]]
        x = np.arange(len(stabilities))
        try: # Linear regression slope
            slope = np.polyfit(x, stabilities, 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, ValueError): # Fallback
            return (stabilities[-1] - stabilities[0]) / (len(stabilities) - 1) if len(stabilities) > 1 else 0.0

    def _apply_stability_influence(self) -> None:
        # Assumes called with lock held
        if len(self.state.memory) < 5: return
        recent = list(self.state.memory)[-5:]
        avg_state = np.mean(recent, axis=0); norm_avg = np.linalg.norm(avg_state)
        if norm_avg < 1e-10: return
        avg_state /= norm_avg

        # Coherence calculation (more robust)
        sims = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                 norm1=np.linalg.norm(recent[i]); norm2=np.linalg.norm(recent[j])
                 if norm1 > 1e-10 and norm2 > 1e-10: sims.append(np.dot(recent[i]/norm1, recent[j]/norm2))
        coherence = np.mean(sims) if sims else 0.0

        if coherence > 0.75: # Higher coherence threshold
            influence = 0.1 * coherence # Scale influence by coherence
            new_state = (1 - influence) * self.state.current + influence * avg_state
            norm_new = np.linalg.norm(new_state); new_state /= norm_new if norm_new > 1e-10 else 1.0
            self.state.update(new_state) # update calls copy internally

    # --- Resonance Functions (_apply_linear_resonance, etc.) ---
    # These should be safe to call without the main lock if they only read dna/resonance_matrix
    # Note: Made minor adjustments for normalization and clarity
    def _apply_linear_resonance(self, data: np.ndarray) -> np.ndarray:
        result = np.dot(self.resonance_matrix, data)
        dna_similarity = np.dot(data, self.dna.encoding)
        result += 0.1 * dna_similarity * self.dna.encoding # Add DNA influence
        norm = np.linalg.norm(result); return result / norm if norm > 1e-10 else result

    def _apply_nonlinear_resonance(self, data: np.ndarray) -> np.ndarray:
        linear_result = np.dot(self.resonance_matrix, data)
        activated = np.tanh(linear_result * 1.7)
        result = 0.7 * activated + 0.3 * linear_result # Skip connection
        dna_similarity = np.dot(data, self.dna.encoding)
        result += 0.15 * dna_similarity * self.dna.encoding
        norm = np.linalg.norm(result); return result / norm if norm > 1e-10 else result

    def _apply_quantum_resonance(self, data: np.ndarray) -> np.ndarray:
        base_result = np.dot(self.resonance_matrix, data)
        noise_amp = 0.03 * (1.0 - self.state.stability) # More noise if unstable
        noise = np.random.randn(self.dimension) * noise_amp
        phase = np.random.uniform(0, 2*np.pi, size=self.dimension)
        complex_noise = noise * np.exp(1j * phase)
        noisy_result = base_result + np.real(complex_noise) # Add real part of complex noise
        result = 0.7 * noisy_result + 0.3 * data # Mix with original
        dna_similarity = np.dot(data, self.dna.encoding)
        phase_influence = np.exp(1j * dna_similarity * np.pi)
        result += 0.1 * np.real(phase_influence) * self.dna.encoding
        norm = np.linalg.norm(result); return result / norm if norm > 1e-10 else result

    def _apply_chaotic_resonance(self, data: np.ndarray) -> np.ndarray:
        base_result = np.dot(self.resonance_matrix, data)
        chaos_r = np.clip(3.57 + np.mean(np.abs(data)) * 0.4, 3.7, 3.99) # Logistic map parameter
        chaos_perturbation = np.zeros_like(base_result); x = 0.5
        for i in range(self.dimension): x = chaos_r * x * (1 - x); chaos_perturbation[i] = x
        chaos_perturbation = (chaos_perturbation - 0.5) * 0.2 # Scale and center
        result = base_result + chaos_perturbation
        result = np.tanh(result * 1.5) # Apply tanh activation
        dna_similarity = np.dot(data, self.dna.encoding)
        result += 0.1 * dna_similarity * self.dna.encoding
        norm = np.linalg.norm(result); return result / norm if norm > 1e-10 else result

    def _apply_topological_resonance(self, data: np.ndarray) -> np.ndarray:
        base_result = np.dot(self.resonance_matrix, data)
        result = base_result.copy(); n = self.dimension
        # Simple ring topology influence
        for scale_pow in range(5):
            scale = 2 ** (scale_pow + 1)
            rolled_fwd = np.roll(base_result, scale)
            rolled_bwd = np.roll(base_result, -scale)
            diff_fwd = rolled_fwd - base_result
            diff_bwd = rolled_bwd - base_result
            weight_fwd = np.exp(-np.abs(diff_fwd) * 2)
            weight_bwd = np.exp(-np.abs(diff_bwd) * 2)
            result += 0.005 * (weight_fwd * diff_fwd + weight_bwd * diff_bwd) # Smaller influence factor

        # DNA connectivity influence (simplified region mapping)
        n_regions = min(32, n // 8)
        if n_regions > 1:
            region_size = n // n_regions
            region_means = [np.mean(data[i*region_size : (i+1)*region_size]) for i in range(n_regions)]
            for i in range(n_regions):
                dna_idx = i % len(self.dna.connectivity_pattern)
                target_region = int(self.dna.connectivity_pattern[dna_idx] * n_regions) % n_regions
                if target_region != i:
                    src_mean = region_means[i]; conn_strength = self.dna.connectivity_pattern[dna_idx] * 0.1
                    tgt_start = target_region * region_size; tgt_end = (target_region + 1) * region_size
                    result[tgt_start:tgt_end] += conn_strength * src_mean

        result = np.tanh(result * 1.6) # Activation
        norm = np.linalg.norm(result); return result / norm if norm > 1e-10 else result

    def _apply_hybrid_resonance(self, data: np.ndarray) -> np.ndarray:
        linear_r = self._apply_linear_resonance(data); nonlinear_r = self._apply_nonlinear_resonance(data)
        quantum_r = self._apply_quantum_resonance(data); chaotic_r = self._apply_chaotic_resonance(data)
        topological_r = self._apply_topological_resonance(data)
        # Combine using DNA weights (ensure weights sum ~1)
        total_weight = self.dna.linear_weight + self.dna.nonlinear_weight + self.dna.quantum_weight + self.dna.chaotic_weight + self.dna.topological_weight
        if total_weight < 1e-6: total_weight=1.0 # Avoid division by zero
        result = (self.dna.linear_weight * linear_r + self.dna.nonlinear_weight * nonlinear_r +
                  self.dna.quantum_weight * quantum_r + self.dna.chaotic_weight * chaotic_r +
                  self.dna.topological_weight * topological_r) / total_weight

        # Knowledge influence based on similarity
        if self.knowledge_base:
            similarities = np.array([np.dot(data, k) for k in self.knowledge_base])
            max_idx = np.argmax(similarities); max_sim = similarities[max_idx]
            if max_sim > 0.4: # Similarity threshold
                 knowledge = self.knowledge_base[max_idx] # Access deque element by index
                 result += 0.15 * max_sim * knowledge

        norm = np.linalg.norm(result); return result / norm if norm > 1e-10 else result

    def get_status(self) -> Dict[str, Any]:
        # Acquire lock to safely read state and metrics
        with self.lock:
            state_summary = {
                "energy": self.state.energy, "stability": self.state.stability, "coherence": self.state.coherence,
                "memory_size": len(self.state.memory), "attractor_count": len(self.state.attractors)
            }
            # Create copies of mutable dicts/lists if necessary for external use
            dna_summary = {
                 "linear_weight": self.dna.linear_weight, "nonlinear_weight": self.dna.nonlinear_weight,
                 "quantum_weight": self.dna.quantum_weight, "chaotic_weight": self.dna.chaotic_weight,
                 "topological_weight": self.dna.topological_weight, "mutation_rate": self.dna.mutation_rate
            }
            metrics_copy = self.metrics.copy()

        return {
            "resonance_mode": self.resonance_mode.name, "dna": dna_summary, "state": state_summary,
            "knowledge_base_size": len(self.knowledge_base), "metrics": metrics_copy,
            "evolution_active": self.evolution_active, "auto_tune": self.auto_tune
        }

# ===================================
# Pattern Processor
# ===================================
class PatternProcessor:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension; self.patterns = {}; self.pattern_counter = 0
        self.max_patterns = 10000
        self.correlation_threshold = 0.6; self.novelty_threshold = 0.4; self.importance_threshold = 0.3
        self.metrics = {"patterns_detected": 0, "duplicate_patterns": 0, "pattern_quality": 0.7}
        self.logger = logging.getLogger(f"PatternProc_{id(self)}")

    def detect_patterns(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> List[Pattern]:
        if metadata is None: metadata = {}
        data = data.reshape(1, -1) if len(data.shape) == 1 else data # Ensure 2D for processing
        data = data[:, :self.dimension] # Ensure correct dimension

        detected_patterns = []
        detected_patterns.extend(self._detect_structural_patterns(data))
        if 'sequence' in metadata or 'time_series' in metadata or data.shape[0] > 1: # Check if data looks sequential
            detected_patterns.extend(self._detect_sequential_patterns(data))
        detected_patterns.extend(self._detect_anomalies(data))

        self.metrics["patterns_detected"] += len(detected_patterns)
        unique_patterns = []; current_pattern_ids = set(self.patterns.keys())
        # More efficient duplicate check: use a set of hashes or simplified vectors?
        # For now, stick to similarity check but optimize maybe
        existing_patterns_list = list(self.patterns.values()) # Avoid iterating dict values repeatedly

        for pattern in detected_patterns:
            is_duplicate = False
            # Only check against patterns not already marked for removal if pruning is implemented
            for existing in existing_patterns_list:
                if pattern.similarity(existing) > self.correlation_threshold:
                    is_duplicate = True; self.metrics["duplicate_patterns"] += 1; break
            if not is_duplicate:
                unique_patterns.append(pattern)
                # Add to existing list for checks against subsequent new patterns in this batch
                existing_patterns_list.append(pattern)

        # Add unique patterns, potentially pruning old ones if max_patterns is reached
        added_count = 0
        for pattern in unique_patterns:
            if len(self.patterns) >= self.max_patterns:
                 # Simple pruning: remove oldest pattern
                 try: oldest_id = next(iter(self.patterns)); del self.patterns[oldest_id]
                 except StopIteration: pass # Should not happen if len >= max
            if pattern.id not in self.patterns: # Ensure ID is unique just in case
                self.patterns[pattern.id] = pattern
                added_count += 1

        # Update quality metric
        if detected_patterns:
            avg_confidence = np.mean([p.confidence for p in detected_patterns]) if detected_patterns else 0.0
            unique_ratio = len(unique_patterns) / len(detected_patterns) if detected_patterns else 0.0
            self.metrics["pattern_quality"] = 0.6 * self.metrics["pattern_quality"] + 0.4 * (0.5 * avg_confidence + 0.5 * unique_ratio) # Smoothing

        # self.logger.debug(f"Detected {len(detected_patterns)}, added {added_count} unique patterns.")
        return unique_patterns # Return only the newly added unique ones

    def _detect_structural_patterns(self, data: np.ndarray) -> List[Pattern]:
        patterns = []
        try:
            # Center data before SVD
            mean = np.mean(data, axis=0); centered_data = data - mean
            # Limit SVD components if data dimension is high
            k = min(data.shape[0], data.shape[1], 50) # Max 50 components
            U, s, Vh = np.linalg.svd(centered_data, full_matrices=False)
            explained_variance = (s**2) / np.sum(s**2) if np.sum(s**2) > 0 else np.zeros_like(s)

            # Find significant components (e.g., explaining > 5% variance or elbow method)
            significant_indices = np.where(explained_variance > 0.05)[0]
            significant_indices = significant_indices[:min(len(significant_indices), 10)] # Limit number of structural patterns

            for i in significant_indices[:k]: # Ensure index is within calculated Vh
                component = Vh[i, :] # Principal component vector
                component = self._ensure_dimension(component) # Ensure correct dimension
                confidence = float(explained_variance[i])
                if confidence < 0.01: continue # Skip very low confidence

                self.pattern_counter += 1; pattern_id = f"struct_{int(time.time())}_{self.pattern_counter}"
                pattern = Pattern(id=pattern_id, type="STRUCTURAL", vector=component, confidence=confidence,
                                  metadata={"component_index": int(i), "explained_variance": confidence})
                patterns.append(pattern)
        except (np.linalg.LinAlgError, ValueError) as e:
            self.logger.warning(f"SVD failed in structural pattern detection: {e}")
        return patterns

    def _detect_sequential_patterns(self, data: np.ndarray) -> List[Pattern]:
         patterns = []; n_series, n_points = data.shape
         if n_points < 10: return [] # Need sufficient points

         # Process first few series for autocorrelation patterns
         for i in range(min(n_series, 5)):
             series = data[i, :] - np.mean(data[i, :]) # Detrend simple mean
             if np.std(series) < 1e-6: continue # Skip flat series
             try:
                 autocorr = np.correlate(series, series, mode='full')
                 autocorr = autocorr[len(autocorr)//2:] # Positive lags
                 autocorr /= autocorr[0] # Normalize
                 # Find peaks (simple method)
                 peaks = []
                 min_period = 3; max_period = n_points // 2
                 for lag in range(min_period, min(len(autocorr)-1, max_period)):
                      if autocorr[lag] > 0.3 and autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                           peaks.append((lag, autocorr[lag]))
                 # Create patterns for strongest peaks
                 peaks.sort(key=lambda x: x[1], reverse=True)
                 for period, strength in peaks[:3]: # Max 3 seq patterns per series
                      subsequence = np.zeros(self.dimension)
                      # Represent pattern by averaging segments? Or just use first segment?
                      # Let's use average of first few segments
                      num_segments = min(5, n_points // period)
                      if num_segments > 0:
                           avg_segment = np.mean([series[k*period:(k+1)*period] for k in range(num_segments)], axis=0)
                           len_segment = len(avg_segment)
                           subsequence[:min(len_segment, self.dimension)] = avg_segment[:min(len_segment, self.dimension)]
                      else: # Fallback: use first segment if too short
                           subsequence[:min(period, self.dimension)] = series[:min(period, self.dimension)]

                      self.pattern_counter += 1; pattern_id = f"seq_{int(time.time())}_{self.pattern_counter}"
                      pattern = Pattern(id=pattern_id, type="SEQUENTIAL", vector=subsequence / (np.linalg.norm(subsequence) or 1.0),
                                        confidence=float(strength), metadata={"periodicity": int(period), "strength": float(strength), "series_index": int(i)})
                      patterns.append(pattern)
             except Exception as e:
                  self.logger.warning(f"Error detecting sequential patterns for series {i}: {e}")
         return patterns

    def _detect_anomalies(self, data: np.ndarray) -> List[Pattern]:
        patterns = []; n_series, n_points = data.shape
        if n_points < 5: return []

        for i in range(n_series):
            series = data[i, :]
            mean = np.mean(series); std = np.std(series)
            if std < 1e-6: continue # Skip flat series

            z_scores = (series - mean) / std
            outlier_indices = np.where(np.abs(z_scores) > 3.0)[0] # Threshold z > 3

            for idx in outlier_indices[:10]: # Limit anomalies per series
                anomaly_vec = np.zeros(self.dimension)
                # Simple representation: index, value, z-score
                anomaly_vec[0] = i; anomaly_vec[1] = idx; anomaly_vec[2] = series[idx]; anomaly_vec[3] = z_scores[idx]
                self.pattern_counter += 1; pattern_id = f"anomaly_{int(time.time())}_{self.pattern_counter}"
                confidence = float(np.clip(0.5 + 0.1 * (np.abs(z_scores[idx]) - 3.0), 0.5, 0.95))
                pattern = Pattern(id=pattern_id, type="ANOMALY", vector=anomaly_vec, confidence=confidence,
                                  metadata={"series_index": int(i), "position": int(idx), "value": float(series[idx]), "z_score": float(z_scores[idx])})
                patterns.append(pattern)
        return patterns

    def get_patterns(self, limit: int = 100) -> List[Pattern]:
        patterns = list(self.patterns.values())
        # Sort by confidence, then timestamp (newest first)
        patterns.sort(key=lambda p: (p.confidence, p.timestamp), reverse=True)
        return patterns[:limit]

    # Helper needed within PatternProcessor
    def _ensure_dimension(self, vector: np.ndarray) -> np.ndarray:
         current_dim = len(vector)
         if current_dim == self.dimension: return vector
         elif current_dim > self.dimension: return vector[:self.dimension]
         else:
             padded = np.zeros(self.dimension, dtype=vector.dtype); padded[:current_dim] = vector; return padded

# ===================================
# Insight Generator
# ===================================
class InsightGenerator:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension; self.insights = {}; self.patterns = {} # Store patterns used by insights?
        self.insight_counter = 0; self.max_insights = 1000
        self.metrics = {"insights_generated": 0, "insight_quality": 0.7, "novel_insights": 0, "insight_importance": 0.5}
        self.logger = logging.getLogger(f"InsightGen_{id(self)}")
        self._combinations_cache = {} # Cache for combinations helper

    def add_pattern(self, pattern: Pattern) -> None:
        # Maybe store patterns temporarily for insight generation pass?
        self.patterns[pattern.id] = pattern # Temporarily store? Or rely on caller passing them?

    def generate_insights(self, current_patterns: List[Pattern]) -> List[Insight]:
        # Add new patterns to internal cache if needed
        for p in current_patterns: self.patterns[p.id] = p
        if len(self.patterns) < 2: return [] # Need patterns to generate insights

        all_patterns = list(self.patterns.values()) # Use all available patterns
        insights = []
        insights.extend(self._generate_correlation_insights(all_patterns))
        insights.extend(self._generate_predictive_insights(all_patterns))
        insights.extend(self._generate_integration_insights(all_patterns))

        # Filter insights based on novelty/redundancy with existing insights
        new_unique_insights = []
        existing_insight_vectors = [ins.vector for ins in self.insights.values()] # Get existing vectors

        for insight in insights:
             is_novel = True
             if existing_insight_vectors:
                  similarities = [self._vector_similarity(insight.vector, v) for v in existing_insight_vectors]
                  if similarities and np.max(similarities) > 0.9: # High similarity threshold
                       is_novel = False
             if is_novel:
                  new_unique_insights.append(insight)
                  if len(self.insights) >= self.max_insights:
                       # Pruning: Remove oldest or lowest importance insight
                       try: oldest_id = next(iter(self.insights)); del self.insights[oldest_id]
                       except StopIteration: pass
                  self.insights[insight.id] = insight
                  existing_insight_vectors.append(insight.vector) # Add to vectors for checks in this batch
                  self.metrics["insights_generated"] += 1
                  if insight.novelty > 0.7: self.metrics["novel_insights"] += 1

        # Update metrics
        if new_unique_insights:
             quality = np.mean([i.confidence for i in new_unique_insights])
             importance = np.mean([i.importance for i in new_unique_insights])
             self.metrics["insight_quality"] = 0.7 * self.metrics["insight_quality"] + 0.3 * quality
             self.metrics["insight_importance"] = 0.7 * self.metrics["insight_importance"] + 0.3 * importance

        # self.logger.debug(f"Generated {len(insights)}, added {len(new_unique_insights)} unique insights.")
        # Cleanup pattern cache? Or keep it? Keep for now.
        return new_unique_insights

    def _generate_correlation_insights(self, patterns: List[Pattern]) -> List[Insight]:
        insights = []; n = len(patterns)
        if n < 2: return []
        try:
             # Efficiently calculate upper triangle of correlation matrix
             corr_matrix_upper = {} # Using dict for sparse storage if needed
             vecs = [p.vector for p in patterns]
             for i, j in self._combinations(range(n), 2):
                  corr = self._calculate_correlation(vecs[i], vecs[j])
                  if abs(corr) > 0.65: # Correlation threshold
                      corr_matrix_upper[(i,j)] = corr

             # Simple clustering: find highly correlated pairs/triplets
             processed = set()
             for (i,j), corr_ij in corr_matrix_upper.items():
                 if i in processed or j in processed: continue
                 group = {i, j}
                 avg_corr = corr_ij
                 # Check for a third highly correlated member (triplet)
                 best_k = -1; best_corr_k = 0
                 for k in range(n):
                      if k == i or k == j: continue
                      corr_ik = corr_matrix_upper.get(tuple(sorted((i,k))), 0)
                      corr_jk = corr_matrix_upper.get(tuple(sorted((j,k))), 0)
                      if abs(corr_ik) > 0.65 and abs(corr_jk) > 0.65:
                           # Found a triplet, add k
                           avg_corr = (abs(corr_ij) + abs(corr_ik) + abs(corr_jk)) / 3
                           group.add(k)
                           break # Take first triplet found for simplicity

                 if len(group) >= 2:
                      group_list = list(group)
                      group_patterns = [patterns[idx] for idx in group_list]
                      pattern_ids = [p.id for p in group_patterns]
                      avg_vector = np.mean([p.vector for p in group_patterns], axis=0)
                      avg_vector /= (np.linalg.norm(avg_vector) or 1.0)
                      avg_confidence = np.mean([p.confidence for p in group_patterns])
                      description = f"High correlation detected ({'positive' if avg_corr > 0 else 'negative'}) between {len(group)} patterns: {[p.type for p in group_patterns]}"

                      self.insight_counter += 1; insight_id = f"corr_{int(time.time())}_{self.insight_counter}"
                      insight = Insight(id=insight_id, type="CORRELATION", patterns=pattern_ids, vector=avg_vector, description=description,
                                        confidence=avg_confidence * abs(avg_corr), # Modulate confidence by corr strength
                                        importance=0.4 + 0.1 * len(group), novelty=max(0.1, 1.0 - abs(avg_corr)), # Higher corr = less novel?
                                        metadata={"avg_correlation": float(avg_corr), "pattern_types": [p.type for p in group_patterns]})
                      insights.append(insight)
                      processed.update(group) # Mark group members as processed

        except Exception as e: self.logger.warning(f"Error generating correlation insights: {e}")
        return insights

    def _generate_predictive_insights(self, patterns: List[Pattern]) -> List[Insight]:
        insights = []
        seq_patterns = [p for p in patterns if p.type == "SEQUENTIAL" and 'periodicity' in p.metadata]
        for p in seq_patterns:
             period = p.metadata.get('periodicity', 0)
             strength = p.metadata.get('strength', 0.0)
             if period > 1 and strength > 0.4: # Thresholds
                  self.insight_counter += 1; insight_id = f"pred_{int(time.time())}_{self.insight_counter}"
                  description = f"Potential predictive pattern with period {period} (strength {strength:.2f})"
                  insight = Insight(id=insight_id, type="PREDICTION", patterns=[p.id], vector=p.vector, description=description,
                                    confidence=p.confidence * strength, importance=0.5 + 0.2 * strength, novelty=0.6,
                                    metadata=p.metadata)
                  insights.append(insight)
        return insights

    def _generate_integration_insights(self, patterns: List[Pattern]) -> List[Insight]:
        insights = []
        if len(patterns) < 3: return []
        # Try integrating diverse patterns (different types, maybe some anomalies)
        anomalies = [p for p in patterns if p.type == "ANOMALY" and p.confidence > 0.6]
        structurals = [p for p in patterns if p.type == "STRUCTURAL" and p.confidence > 0.5]
        sequentials = [p for p in patterns if p.type == "SEQUENTIAL" and p.confidence > 0.5]

        # Example: Integrate 1 anomaly with 2 structural patterns
        if anomalies and len(structurals) >= 2:
             combo = [anomalies[0]] + self._random_sample(structurals, 2) # Use random sample helper
             pattern_ids = [p.id for p in combo]; vectors = [p.vector for p in combo]
             weights = np.array([p.confidence for p in combo]); weights /= (np.sum(weights) or 1.0)
             integrated_vector = np.sum([w * v for w, v in zip(weights, vectors)], axis=0)
             integrated_vector /= (np.linalg.norm(integrated_vector) or 1.0)
             avg_confidence = np.mean([p.confidence for p in combo])
             description = f"Integration insight: Anomaly '{anomalies[0].metadata.get('value', 'N/A')}' potentially related to structural patterns."

             self.insight_counter += 1; insight_id = f"integ_{int(time.time())}_{self.insight_counter}"
             insight = Insight(id=insight_id, type="INTEGRATION", patterns=pattern_ids, vector=integrated_vector, description=description,
                               confidence=avg_confidence * 0.8, # Lower confidence for speculative integration
                               importance=0.6, novelty=0.75, # Integration often novel
                               metadata={"integrated_types": [p.type for p in combo]})
             insights.append(insight)

        # Add more integration strategies here...
        return insights

    def _calculate_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
         # Pearson correlation coefficient
         try:
             # Ensure compatible shapes and variance
             v1 = vec1[:min(len(vec1), len(vec2))]; v2 = vec2[:min(len(vec1), len(vec2))]
             if np.std(v1) < 1e-6 or np.std(v2) < 1e-6: return 0.0 # Avoid NaN if no variance
             corr = np.corrcoef(v1, v2)[0, 1]
             return float(np.nan_to_num(corr)) # Handle potential NaN
         except ValueError: return 0.0 # Handle shape mismatches etc.

    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
         # Cosine similarity
         norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
         if norm1 < 1e-10 or norm2 < 1e-10: return 0.0
         return float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0))

    # --- Helper methods for combinations and random_sample ---
    # (Copy from previous version or use itertools/random if available)
    @staticmethod
    def _combinations(items, r):
        # Simple combination generator (replacement for itertools.combinations)
        n = len(items); if r > n: return
        indices = list(range(r))
        yield tuple(indices) # Yield indices, not items
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r: break
            else: return
            indices[i] += 1
            for j in range(i+1, r): indices[j] = indices[j-1] + 1
            yield tuple(indices)

    @staticmethod
    def _random_sample(items, k):
        n = len(items); k = min(k, n)
        if k == 0: return []
        indices = np.random.choice(n, k, replace=False)
        return [items[i] for i in indices]

    def get_all_insights(self) -> List[Dict[str, Any]]:
         return [ { "id": i.id, "type": i.type, "description": i.description[:100], # Truncate desc
                    "confidence": i.confidence, "importance": i.importance, "novelty": i.novelty }
                  for i in self.insights.values() ]


# ===================================
# Perspective Generator
# ===================================
class PerspectiveGenerator:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension; self.perspectives = {}; self.insights = {}
        self.perspective_counter = 0; self.max_perspectives = 100
        self.metrics = {"perspectives_generated": 0, "perspective_impact": 0.0, "perspective_coherence": 0.0, "perspective_novelty": 0.0}
        self.logger = logging.getLogger(f"PerspGen_{id(self)}")

    def add_insight(self, insight: Insight) -> None:
        self.insights[insight.id] = insight

    def generate_perspectives(self, new_insights: List[Insight] = None) -> List[Perspective]:
        if new_insights:
             for insight in new_insights: self.insights[insight.id] = insight
        if len(self.insights) < 3: return [] # Need enough insights

        all_insights = list(self.insights.values())
        insight_groups = self._group_insights(all_insights) # Group similar insights
        perspectives = []

        # Cache perspective vectors to calculate novelty against other groups
        group_perspective_vectors = {}

        for i, group_ids in enumerate(insight_groups):
            if len(group_ids) < 2: continue # Need multiple insights for a perspective
            group_insights = [self.insights[id] for id in group_ids]
            insight_ids = [insight.id for insight in group_insights]
            vectors = [insight.vector for insight in group_insights]
            importances = np.array([insight.importance for insight in group_insights])
            weights = importances / np.sum(importances) if np.sum(importances) > 0 else np.ones(len(importances)) / len(importances)

            perspective_vector = np.sum([w * v for w, v in zip(weights, vectors)], axis=0)
            norm = np.linalg.norm(perspective_vector); perspective_vector /= norm if norm > 1e-10 else 1.0
            group_perspective_vectors[i] = perspective_vector # Cache vector for novelty calc

            # Coherence
            sims = [self._vector_similarity(vectors[idx1], vectors[idx2]) for idx1, idx2 in self._combinations(range(len(vectors)), 2)]
            coherence = np.mean(sims) if sims else 0.5
            avg_importance = np.mean(importances)

            # Description
            types = [insight.type for insight in group_insights]; type_counts = {t: types.count(t) for t in set(types)}
            type_str = ", ".join(f"{count} {t}" for t, count in type_counts.items())
            description = f"Perspective integrating {len(group_insights)} insights ({type_str})"

            # Novelty and Impact will be calculated after all group vectors are known
            perspectives.append({
                 "group_index": i, "insight_ids": insight_ids, "vector": perspective_vector, "strength": avg_importance,
                 "coherence": coherence, "description": description, "metadata": {"types": list(type_counts.keys()), "count": len(group_insights)}
            })

        # Calculate novelty and impact, then create Perspective objects
        final_perspectives = []
        for p_data in perspectives:
             group_idx = p_data["group_index"]
             perspective_vector = p_data["vector"]
             # Novelty: Compare to other group vectors
             other_sims = []
             for other_idx, other_vec in group_perspective_vectors.items():
                  if group_idx == other_idx: continue
                  other_sims.append(self._vector_similarity(perspective_vector, other_vec))
             avg_other_sim = np.mean(other_sims) if other_sims else 0.0
             novelty = max(0.0, 1.0 - avg_other_sim) # Higher novelty if dissimilar to others

             impact = 0.4 * p_data["coherence"] + 0.4 * novelty + 0.2 * p_data["strength"] # Weighted impact score

             self.perspective_counter += 1; perspective_id = f"persp_{int(time.time())}_{self.perspective_counter}"
             perspective = Perspective(id=perspective_id, insight_ids=p_data["insight_ids"], vector=perspective_vector,
                                       strength=p_data["strength"], coherence=p_data["coherence"], novelty=novelty,
                                       impact=impact, description=p_data["description"], metadata=p_data["metadata"])

             # Add/Prune perspectives store
             if len(self.perspectives) >= self.max_perspectives:
                  # Pruning: Remove lowest impact perspective
                  lowest_impact_id = min(self.perspectives, key=lambda k: self.perspectives[k].impact, default=None)
                  if lowest_impact_id: del self.perspectives[lowest_impact_id]
             if perspective.id not in self.perspectives: # Check just in case
                 self.perspectives[perspective.id] = perspective
                 final_perspectives.append(perspective)
                 self.metrics["perspectives_generated"] += 1

        # Update metrics
        if final_perspectives:
             self.metrics["perspective_impact"] = np.mean([p.impact for p in final_perspectives])
             self.metrics["perspective_coherence"] = np.mean([p.coherence for p in final_perspectives])
             self.metrics["perspective_novelty"] = np.mean([p.novelty for p in final_perspectives])

        # self.logger.debug(f"Generated {len(perspectives)}, added {len(final_perspectives)} perspectives.")
        return final_perspectives

    def _group_insights(self, insights: List[Insight]) -> List[List[str]]:
        n = len(insights); if n < 2: return [[i.id] for i in insights]
        adj = {i.id: [] for i in insights}
        insight_map = {i: insight for i, insight in enumerate(insights)} # Map index to insight

        # Build adjacency list based on similarity
        threshold = 0.65 # Grouping threshold
        for i, j in self._combinations(range(n), 2):
             sim = self._vector_similarity(insight_map[i].vector, insight_map[j].vector)
             if sim > threshold:
                  adj[insight_map[i].id].append(insight_map[j].id)
                  adj[insight_map[j].id].append(insight_map[i].id)

        # Find connected components (groups) using BFS
        groups = []; visited = set()
        for i_id in adj:
            if i_id not in visited:
                component = []; queue = deque([i_id]); visited.add(i_id)
                while queue:
                    current_id = queue.popleft(); component.append(current_id)
                    for neighbor_id in adj[current_id]:
                        if neighbor_id not in visited: visited.add(neighbor_id); queue.append(neighbor_id)
                if component: groups.append(component)
        return groups

    # --- Helper methods ---
    # (Copy from InsightGenerator or use shared utility)
    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
         norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
         if norm1 < 1e-10 or norm2 < 1e-10: return 0.0
         return float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0))

    @staticmethod
    def _combinations(items, r):
        n = len(items); if r > n: return
        indices = list(range(r))
        yield tuple(indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r: break
            else: return
            indices[i] += 1
            for j in range(i+1, r): indices[j] = indices[j-1] + 1
            yield tuple(indices)

    def get_all_perspectives(self) -> List[Dict[str, Any]]:
         # Return summary using the Perspective's to_dict method
         return [p.to_dict() for p in self.perspectives.values()]

# ===================================
# Kaleidoscope Engine
# ===================================
class KaleidoscopeEngine:
    def __init__(self, dimension: int = 512, resonance_mode: ResonanceMode = ResonanceMode.HYBRID):
        self.dimension = dimension; self.resonance_mode = resonance_mode
        self.core = SuperNodeCore(dimension=dimension, resonance_mode=resonance_mode)
        self.pattern_processor = PatternProcessor(dimension=dimension)
        self.insight_generator = InsightGenerator(dimension=dimension)
        self.perspective_generator = PerspectiveGenerator(dimension=dimension)
        self.string_cube = StringCube(dimension=3, resolution=32) # Cube dimension fixed at 3
        self.nodes = {} # id -> ConsciousNode
        self.metrics = {"processing_count": 0, "pattern_count": 0, "insight_count": 0,
                        "perspective_count": 0, "node_count": 0}
        self.core.start() # Start the core's background evolution
        self.logger = logging.getLogger("KaleidoscopeEngine")
        self.processing_lock = threading.Lock() # Lock for main processing pipeline if needed

    def create_node(self, features: np.ndarray, position: Optional[np.ndarray] = None, energy: float = 0.5, stability: float = 0.8) -> str:
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        if position is None: position = np.random.uniform(-1, 1, size=3) # Position always 3D for cube
        else: position = np.clip(position[:3], -1.0, 1.0) # Ensure 3D and clipped
        features = self.core._ensure_dimension(features) # Use core's helper

        node = ConsciousNode(id=node_id, position=position, energy=energy, stability=stability, features=features)
        with self.processing_lock: # Protect shared nodes dict and cube
            self.nodes[node_id] = node
            self.string_cube.add_node(node)
            self.metrics["node_count"] = len(self.nodes) # Update count accurately
        return node_id

    def connect_nodes(self, node1_id: str, node2_id: str, strength: Optional[float] = None) -> bool:
         with self.processing_lock: # Protect nodes dict and cube updates
            if node1_id not in self.nodes or node2_id not in self.nodes: return False
            node1 = self.nodes[node1_id]; node2 = self.nodes[node2_id]
            if strength is None: strength = node1.calculate_affinity(node2)
            strength = np.clip(strength, 0.0, 1.0) # Ensure valid strength
            node1.connections[node2_id] = strength
            node2.connections[node1_id] = strength # Assuming undirected graph
            # Update string cube tension (might be slow, consider batching or async?)
            self.string_cube.update_tension(self.nodes)
         return True

    def process_input(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        if metadata is None: metadata = {}

        with self.processing_lock: # Lock the entire processing pipeline? Or finer-grained?
            data = self.core._ensure_dimension(data.flatten()) # Prepare data
            processed_data = self.core.process_input(data) # Process through core

            # Pattern detection using core's output
            patterns = self.pattern_processor.detect_patterns(processed_data, metadata)
            # Generate insights using newly detected patterns
            insights = self.insight_generator.generate_insights(patterns)
            # Generate perspectives using newly generated insights
            perspectives = self.perspective_generator.generate_perspectives(insights)

            # Absorb knowledge (e.g., from new perspectives) back into the core
            if perspectives:
                # Use perspective vectors, potentially weighted by impact
                knowledge_vecs = [p.vector * p.impact for p in perspectives if p.impact > 0.3]
                if knowledge_vecs:
                     # Absorb average or most impactful? Let's absorb average.
                     avg_knowledge = np.mean(knowledge_vecs, axis=0)
                     self.core.absorb_knowledge(avg_knowledge)

            # Run simulation step (includes cube updates, node energy decay etc.)
            # Should this be inside or outside the lock? Outside likely better for performance.
            # self.run_simulation_step() # Let's move this outside or make it periodic

            # Update metrics
            self.metrics["processing_count"] += 1
            # Counts are now handled internally by processors/generators
            self.metrics["pattern_count"] = len(self.pattern_processor.patterns)
            self.metrics["insight_count"] = len(self.insight_generator.insights)
            self.metrics["perspective_count"] = len(self.perspective_generator.perspectives)

        # Run simulation step outside the main processing lock
        self.run_simulation_step()

        result = {
            "processing_time": time.time() - start_time,
            "pattern_count": len(patterns), # New patterns this cycle
            "insight_count": len(insights), # New insights this cycle
            "perspective_count": len(perspectives), # New perspectives this cycle
            "patterns": [p.id for p in patterns],
            "insights": [i.id for i in insights],
            "perspectives": [p.id for p in perspectives]
        }
        return result

    def run_simulation_step(self) -> None:
        """Run a single simulation step, updating nodes and cube."""
        with self.processing_lock: # Lock needed for accessing/modifying nodes and cube
            nodes_copy = self.nodes.copy() # Operate on a copy if modification happens

            # Update nodes (energy decay, quantum evolution)
            for node in nodes_copy.values():
                node.update_energy(0.995) # Decay energy
                # Add other per-node updates (e.g., internal state, quantum state evolution)

            # Update string cube tension based on current node states
            self.string_cube.update_tension(nodes_copy)
            # Apply tension effects back to nodes
            self.string_cube.apply_tension_to_nodes(nodes_copy)
            # Evolve the cube's phase grid
            self.string_cube.evolve_quantum_phase_grid()

            # Update the main nodes dictionary with changes from the copy
            self.nodes.update(nodes_copy)

        # Periodically trigger insight/perspective generation based on overall state? (Optional)
        # This could happen less frequently than every step or processing call.
        # if self.metrics["processing_count"] % 20 == 0: # Example: every 20 steps
        #     self.logger.debug("Triggering periodic insight/perspective refresh...")
        #     all_patterns = list(self.pattern_processor.patterns.values())
        #     new_insights = self.insight_generator.generate_insights(all_patterns) # Refresh insights
        #     all_insights = list(self.insight_generator.insights.values())
        #     self.perspective_generator.generate_perspectives(all_insights) # Refresh perspectives


    def get_status(self) -> Dict[str, Any]:
        # Combine status from core and engine metrics
        core_status = self.core.get_status()
        engine_metrics = self.metrics.copy() # Get current metrics
        # Update counts from component stores (more accurate than incremental)
        engine_metrics["node_count"] = len(self.nodes)
        engine_metrics["pattern_count"] = len(self.pattern_processor.patterns)
        engine_metrics["insight_count"] = len(self.insight_generator.insights)
        engine_metrics["perspective_count"] = len(self.perspective_generator.perspectives)
        return {
            "engine_metrics": engine_metrics,
            "core_status": core_status,
            "timestamp": time.time()
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        # Ensure thread safety when accessing shared data
        with self.processing_lock:
            nodes_copy = list(self.nodes.values()) # Get copy of nodes
            patterns_copy = self.pattern_processor.get_patterns(limit=100) # Get sorted patterns
            insights_copy = self.insight_generator.get_all_insights() # Get insight summaries
            perspectives_copy = self.perspective_generator.get_all_perspectives() # Get perspective summaries
            # Calculate tension field based on current cube state
            tension_field = self.string_cube.calculate_scalar_tension_field()

        # Process data outside the lock
        nodes_data = []
        connections_data = set() # Use set to auto-handle duplicates

        for node in nodes_copy:
            grid_pos = self.string_cube._continuous_to_grid(node.position)
            tension_at_node = 0.0
            # Check if grid_pos is valid index for tension_field
            if all(0 <= p < s for p, s in zip(grid_pos, tension_field.shape)):
                 tension_at_node = float(tension_field[grid_pos])

            nodes_data.append({
                'id': node.id, 'position': node.position.tolist(), 'energy': node.energy,
                'stability': node.stability, 'tension': tension_at_node,
                # Add quantum entropy?
                'entropy': node.quantum_state.get_entropy() if node.quantum_state else 0.0
            })
            # Add connections (undirected)
            for conn_id, strength in node.connections.items():
                 # Ensure target node still exists in the copied list (or check against main dict?)
                 if any(n.id == conn_id for n in nodes_copy):
                      conn_key = tuple(sorted((node.id, conn_id)))
                      connections_data.add((*conn_key, strength)) # Add as tuple to set

        # Convert set of connections back to list of dicts
        connections_list = [{'source': src, 'target': tgt, 'strength': s} for src, tgt, s in connections_data]

        # Use data already retrieved outside lock
        patterns_data = [{'id': p.id, 'type': p.type, 'confidence': p.confidence, 'timestamp': p.timestamp} for p in patterns_copy]
        insights_data = insights_copy # Already summaries
        perspectives_data = perspectives_copy # Already summaries

        engine_metrics_copy = self.metrics.copy()
        engine_metrics_copy["node_count"] = len(nodes_copy) # Update with count from copy

        return {
            'nodes': nodes_data, 'connections': connections_list, 'patterns': patterns_data,
            'insights': insights_data, 'perspectives': perspectives_data,
            'metrics': engine_metrics_copy, 'timestamp': time.time()
            # Include cube tension field? Might be large.
            # 'tension_field': tension_field.tolist() # Optional, potentially large data
        }


# ===================================
# Quantum Kaleidoscope System (Top Level)
# ===================================
class QuantumKaleidoscope:
    def __init__(self, dimension: int = 512, data_dir: str = "./data"):
        self.dimension = dimension; self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.engine = KaleidoscopeEngine(dimension=dimension) # Initialize the engine
        # Placeholder encoder/decoder
        self.encoder = lambda x: self._encode_data(x)
        self.decoder = lambda x: self._decode_data(x)
        self.results = {} # Cache for processing results
        self.logger = logging.getLogger("QuantumKaleidoscope") # Use specific logger
        self.auto_gen_active = False; self.auto_gen_thread = None; self.auto_gen_stop_event = threading.Event()

    def start_auto_generation(self, interval: float = 5.0):
        if self.auto_gen_active: self.logger.warning("Auto-generation already active"); return
        self.auto_gen_stop_event.clear(); self.auto_gen_active = True
        self.auto_gen_thread = threading.Thread(target=self._auto_gen_loop, args=(interval,), daemon=True, name="QKAutoGenerate")
        self.auto_gen_thread.start()
        self.logger.info(f"Started auto-generation with interval {interval}s")

    def stop_auto_generation(self):
        if not self.auto_gen_active: self.logger.warning("Auto-generation not active"); return
        self.auto_gen_stop_event.set()
        if self.auto_gen_thread: self.auto_gen_thread.join(timeout=max(1.0, self.engine.core.evolution_interval * 2)) # Wait a bit longer
        self.auto_gen_active = False
        self.logger.info("Stopped auto-generation")

    def _auto_gen_loop(self, interval: float):
        self.logger.info("Auto-generation loop started.")
        while not self.auto_gen_stop_event.is_set():
            try:
                # Generate random features for a new node
                features = np.random.randn(self.dimension); features /= np.linalg.norm(features) if np.linalg.norm(features) > 1e-10 else 1.0
                node_id = self.engine.create_node(features)
                # self.logger.debug(f"Auto-gen created node {node_id}")

                # Connect to a few random existing nodes
                with self.engine.processing_lock: # Need lock to safely access nodes list
                    existing_node_ids = list(self.engine.nodes.keys())
                connect_count = 0
                if len(existing_node_ids) > 1:
                    num_to_connect = min(3, len(existing_node_ids) - 1)
                    # Ensure we don't select the new node itself
                    potential_targets = [nid for nid in existing_node_ids if nid != node_id]
                    if potential_targets:
                         targets = np.random.choice(potential_targets, size=min(num_to_connect, len(potential_targets)), replace=False)
                         for target_id in targets:
                             success = self.engine.connect_nodes(node_id, target_id)
                             if success: connect_count +=1
                # self.logger.debug(f"Auto-gen connected node {node_id} to {connect_count} existing nodes.")

                # Let the engine run its simulation step (already handles internal updates)
                # self.engine.run_simulation_step() # This is now called after process_input

                # Sleep for the interval (use wait for responsiveness)
                self.auto_gen_stop_event.wait(interval)

            except Exception as e:
                self.logger.error(f"Error in auto-generation loop: {e}", exc_info=True)
                self.auto_gen_stop_event.wait(5.0) # Sleep longer on error
        self.logger.info("Auto-generation loop finished.")


    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if metadata is None: metadata = {}
        try:
            data_vector = self.encoder(text)
            text_metadata = {'content_type': 'text', 'text_length': len(text),
                             'original_text': text[:100] + ('...' if len(text) > 100 else '')}
            combined_metadata = {**text_metadata, **metadata} # Combine metadata
            return self.process_data(data_vector, combined_metadata)
        except Exception as e:
            self.logger.error(f"Error encoding or processing text: {e}", exc_info=True)
            return {"error": "Failed to process text", "details": str(e)}

    def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if metadata is None: metadata = {}
        processing_id = f"proc_{uuid.uuid4().hex[:12]}"
        try:
            # Pass data and metadata to the engine's processing pipeline
            result = self.engine.process_input(data, metadata)
            result['processing_id'] = processing_id
            self.results[processing_id] = result # Cache result
            return result
        except Exception as e:
            self.logger.error(f"Error processing data (ID: {processing_id}): {e}", exc_info=True)
            error_result = {"error": "Failed to process data", "details": str(e), "processing_id": processing_id}
            self.results[processing_id] = error_result # Cache error result
            return error_result

    def get_result(self, processing_id: str) -> Optional[Dict[str, Any]]:
        return self.results.get(processing_id)

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        # Return a copy to prevent external modification
        return self.results.copy()

    def get_status(self) -> Dict[str, Any]:
        engine_status = self.engine.get_status() # Engine status includes core status
        return {
            'engine_status': engine_status,
            'auto_gen_active': self.auto_gen_active,
            'results_count': len(self.results),
            'timestamp': time.time()
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        # Delegate to engine
        return self.engine.get_visualization_data()

    # --- Internal Encoder/Decoder Placeholders ---
    def _encode_data(self, text: str) -> np.ndarray:
        """Encode text to vector (placeholder - replace with actual embedding)."""
        # Simple hash/frequency based encoding (same as before)
        vector = np.zeros(self.dimension)
        if not text: return vector
        # Use hash of text to seed random generator for deterministic but unique embeddings
        random_state = np.random.RandomState(seed=int(hashlib.sha256(text.encode()).hexdigest(), 16) & 0xFFFFFFFF)
        vector = random_state.randn(self.dimension) # Generate random vector
        norm = np.linalg.norm(vector); vector /= norm if norm > 1e-10 else 1.0 # Normalize
        # Add simple features
        words = text.split(); num_words = len(words)
        if num_words > 0:
            avg_len = sum(len(w) for w in words) / num_words
            vector[0] = np.clip(avg_len / 15.0, 0, 1) # Feature for avg word length
            vector[1] = np.clip(num_words / 100.0, 0, 1) # Feature for number of words
        return vector

    def _decode_data(self, vector: np.ndarray) -> str:
        """Attempt to decode vector back to text (very approximate placeholder)."""
        # Simple representation based on strongest dimensions
        top_indices = np.argsort(-np.abs(vector))[:5] # Show top 5 dimensions
        dims = [f"d{i}:{vector[i]:.2f}" for i in top_indices]
        return f"[Vector approx: strength={np.linalg.norm(vector):.3f}, top_dims=[{', '.join(dims)}]]"


# --- Main execution block REMOVED from this file ---
# --- All execution now happens via run_system.py ---
