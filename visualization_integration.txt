#!/usr/bin/env python3
"""
Quantum Kaleidoscope Visualization Integration
=============================================

This script implements the integration of the visualization module
with the existing Autonomous Kaleidoscope System implementation.

To use this patch:
1. Place this file in the same directory as your enhanced-quantum-kaleidoscope.py
2. Add an import statement at the top of enhanced-quantum-kaleidoscope.py:
   from visualization_integration import patch_visualization_system
3. Call the patch function right after creating your system instance:
   patch_visualization_system(app.system)
"""

import os
import sys
import time
import math
import json
import random
import logging
import hashlib
import threading
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger("VisualizationIntegration")

class MultiDimensionalProjector:
    """
    Advanced multi-dimensional projection system that creates 3D representations
    of high-dimensional quantum states using various dimensionality reduction techniques.
    """
    
    def __init__(self, dimension=128, method="hybrid"):
        """
        Initialize projector with specified dimension and method.
        
        Args:
            dimension: Dimensionality of the feature space
            method: Projection method ('hybrid', 'spectral', 'force', or 'nonlinear')
        """
        self.dimension = dimension
        self.method = method
        
        # Constants for different projection methods
        self.projection_matrices = self._initialize_projection_matrices()
        self.spectral_basis = self._initialize_spectral_basis()
        self.nonlinear_coefficients = self._initialize_nonlinear_coefficients()
        
        logger.info(f"Initialized {method} projector for {dimension}-dimensional space")
    
    def _initialize_projection_matrices(self) -> List[List[float]]:
        """Initialize deterministic projection matrices for dimensionality reduction."""
        # Use fixed seed for reproducible projections
        rng = random.Random(42)
        
        # Create three orthogonal projection vectors
        # Implementation of Gram-Schmidt process for orthogonalization
        v1 = [rng.uniform(-1, 1) for _ in range(self.dimension)]
        v1_norm = math.sqrt(sum(x*x for x in v1))
        v1 = [x/v1_norm for x in v1]
        
        v2 = [rng.uniform(-1, 1) for _ in range(self.dimension)]
        # Make v2 orthogonal to v1
        dot_prod = sum(a*b for a, b in zip(v1, v2))
        v2 = [v2[i] - dot_prod*v1[i] for i in range(self.dimension)]
        v2_norm = math.sqrt(sum(x*x for x in v2))
        v2 = [x/v2_norm for x in v2]
        
        v3 = [rng.uniform(-1, 1) for _ in range(self.dimension)]
        # Make v3 orthogonal to v1 and v2
        dot_prod1 = sum(a*b for a, b in zip(v1, v3))
        dot_prod2 = sum(a*b for a, b in zip(v2, v3))
        v3 = [v3[i] - dot_prod1*v1[i] - dot_prod2*v2[i] for i in range(self.dimension)]
        v3_norm = math.sqrt(sum(x*x for x in v3))
        v3 = [x/v3_norm for x in v3]
        
        return [v1, v2, v3]
    
    def _initialize_spectral_basis(self) -> List[Tuple[List[float], List[float]]]:
        """Initialize spectral basis functions for advanced projection."""
        basis = []
        rng = random.Random(42)
        
        # Create 3 sets of frequency components for spectral embedding
        for i in range(3):
            # Generate frequencies and phases
            frequencies = [rng.uniform(0.1, 2.0) for _ in range(min(50, self.dimension))]
            phases = [rng.uniform(0, 2*math.pi) for _ in range(min(50, self.dimension))]
            basis.append((frequencies, phases))
        
        return basis
    
    def _initialize_nonlinear_coefficients(self) -> List[List[Tuple[int, int, float]]]:
        """Initialize coefficients for nonlinear projections using pairwise interactions."""
        coefficients = [[], [], []]  # One list per output dimension
        rng = random.Random(42)
        
        # For each output dimension, generate interaction terms
        for dim in range(3):
            # Generate ~50 pairwise interaction terms
            for _ in range(50):
                i = rng.randint(0, self.dimension-1)
                j = rng.randint(0, self.dimension-1)
                coef = rng.uniform(-1, 1)
                coefficients[dim].append((i, j, coef))
        
        return coefficients
    
    def project_features(self, features: List[float], node_id: str) -> List[float]:
        """
        Project a high-dimensional feature vector to 3D space using specified method.
        
        Args:
            features: High-dimensional feature vector
            node_id: Node identifier used for deterministic projections
            
        Returns:
            3D position vector [x, y, z]
        """
        # Ensure features has the right dimensionality
        if len(features) > self.dimension:
            features = features[:self.dimension]
        elif len(features) < self.dimension:
            features = features + [0.0] * (self.dimension - len(features))
        
        # Create a seed from node_id for deterministic but unique projections
        seed_hash = hashlib.md5(str(node_id).encode()).digest()
        seed_value = int.from_bytes(seed_hash[:4], byteorder='little')
        
        # Choose projection method based on configuration
        if self.method == "hybrid":
            # Combine multiple methods with weighted mixture
            linear_pos = self._linear_projection(features)
            spectral_pos = self._spectral_projection(features, seed_value)
            nonlinear_pos = self._nonlinear_projection(features, seed_value)
            
            # Weighted combination
            pos = [
                0.4 * linear_pos[0] + 0.3 * spectral_pos[0] + 0.3 * nonlinear_pos[0],
                0.4 * linear_pos[1] + 0.3 * spectral_pos[1] + 0.3 * nonlinear_pos[1],
                0.4 * linear_pos[2] + 0.3 * spectral_pos[2] + 0.3 * nonlinear_pos[2]
            ]
        elif self.method == "spectral":
            pos = self._spectral_projection(features, seed_value)
        elif self.method == "nonlinear":
            pos = self._nonlinear_projection(features, seed_value)
        else:
            # Default to linear projection
            pos = self._linear_projection(features)
        
        # Scale position to appropriate visualization range
        scaling_factor = 10.0  # Adjust this based on visualization requirements
        pos = [p * scaling_factor for p in pos]
        
        # Add subtle deterministic variation based on node_id
        rng = random.Random(seed_value)
        jitter = 0.5  # Small jitter to prevent perfect overlap
        pos = [p + rng.uniform(-jitter, jitter) for p in pos]
        
        return pos
    
    def _linear_projection(self, features: List[float]) -> List[float]:
        """Linear projection using pre-computed orthogonal projection matrices."""
        # Project features onto the three basis vectors
        pos = [0, 0, 0]
        for i in range(3):
            pos[i] = sum(f * v for f, v in zip(features, self.projection_matrices[i]))
            # Apply nonlinear scaling for better visualization
            pos[i] = math.copysign(math.sqrt(abs(pos[i])), pos[i])
        
        return pos
    
    def _spectral_projection(self, features: List[float], seed: int) -> List[float]:
        """Spectral projection using sinusoidal basis functions."""
        pos = [0, 0, 0]
        rng = random.Random(seed)
        
        # Generate a unique phase shift based on seed
        phase_shift = rng.uniform(0, 2*math.pi)
        
        # For each output dimension
        for i in range(3):
            frequencies, phases = self.spectral_basis[i]
            
            # Combine feature components with spectral weights
            weighted_sum = 0
            for j, feat in enumerate(features[:min(50, len(features))]):
                if j < len(frequencies):
                    # Apply frequency and phase modulation
                    weighted_sum += feat * math.sin(j * frequencies[j] + phases[j] + phase_shift)
            
            pos[i] = weighted_sum
        
        # Normalize to reasonable range
        norm = math.sqrt(sum(p*p for p in pos))
        if norm > 1e-6:
            pos = [p/norm for p in pos]
        
        return pos
    
    def _nonlinear_projection(self, features: List[float], seed: int) -> List[float]:
        """Nonlinear projection using pairwise feature interactions."""
        pos = [0, 0, 0]
        
        # For each output dimension
        for dim in range(3):
            # Linear component
            linear_term = sum(features[i % len(features)] * (0.1 + i/len(features)) 
                             for i in range(min(20, len(features))))
            
            # Nonlinear component using pairwise interactions
            nonlinear_term = 0
            for i, j, coef in self.nonlinear_coefficients[dim]:
                if i < len(features) and j < len(features):
                    # Quadratic interaction term
                    nonlinear_term += coef * features[i] * features[j]
            
            # Combine with nonlinearity
            pos[dim] = linear_term + math.tanh(nonlinear_term)
        
        return pos


class VisualizationAdapter:
    """
    Adapter class to bridge between the Autonomous Kaleidoscope System and
    visualization requirements. Implements the missing get_visualization_data method.
    """
    
    def __init__(self, system, dimension=128, projection_method="hybrid"):
        """
        Initialize the visualization adapter.
        
        Args:
            system: The Autonomous Kaleidoscope System instance to adapt
            dimension: Dimensionality of the feature space
            projection_method: Method for projecting high-dimensional data
        """
        self.system = system
        self.dimension = dimension
        self.projector = MultiDimensionalProjector(dimension, projection_method)
        
        # Cache for visualization data
        self.cache = None
        self.cache_timestamp = 0
        self.cache_ttl = 5.0  # Cache time-to-live in seconds
        
        # Visualization parameters
        self.params = {
            "node_size_scale": 2.0,      # Scale factor for node sizes in visualization
            "edge_thickness_scale": 1.0, # Scale factor for edge thickness
            "min_edge_strength": 0.2,    # Minimum edge strength to display
            "max_nodes": 1000,          # Maximum number of nodes to visualize
            "color_by": "energy",        # Node coloring strategy
            "layout_iterations": 10,     # Number of force-directed layout iterations
        }
        
        logger.info(f"Visualization adapter initialized for {system.__class__.__name__}")
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get visualization data in the format expected by the front-end.
        This method is added to the system to fix the AttributeError.
        
        Returns:
            Dict containing nodes, connections and metadata for visualization
        """
        # Check if we can use cached data
        current_time = time.time()
        if self.cache and current_time - self.cache_timestamp < self.cache_ttl:
            return self.cache
        
        # Extract nodes and connections from the system
        try:
            # Try various ways to access the quantum state data
            nodes_data = self._extract_nodes_from_system()
            connections_data = self._extract_connections_from_system(nodes_data)
            
            # Process data for visualization
            visualization_data = self._prepare_visualization_data(nodes_data, connections_data)
            
            # Cache the result
            self.cache = visualization_data
            self.cache_timestamp = current_time
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            logger.error(traceback.format_exc())
            
            # Return empty visualization data on error
            return {
                "nodes": [],
                "connections": [],
                "timestamp": current_time,
                "error": str(e)
            }
    
    def _extract_nodes_from_system(self) -> Dict[str, Any]:
        """Extract nodes from the system in a flexible way to handle different implementations."""
        # Try different ways to access node data based on observed system structures
        if hasattr(self.system, 'nodes') and isinstance(self.system.nodes, dict):
            # Direct attribute access
            return self.system.nodes
        elif hasattr(self.system, 'engine') and hasattr(self.system.engine, 'nodes'):
            # Access through engine attribute
            return self.system.engine.nodes
        elif hasattr(self.system, 'quantum_engine') and hasattr(self.system.quantum_engine, 'nodes'):
            # Access through quantum_engine attribute
            return self.system.quantum_engine.nodes
        elif hasattr(self.system, 'get_nodes'):
            # Method access
            return self.system.get_nodes()
        elif hasattr(self.system, 'get_quantum_state'):
            # Method access with state dict
            state = self.system.get_quantum_state()
            if isinstance(state, dict) and 'nodes' in state:
                return state['nodes']
        
        # If we get here, we couldn't find nodes
        logger.warning("Could not find nodes in system, returning empty node dict")
        return {}
    
    def _extract_connections_from_system(self, nodes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract connections from the system or derive them from node data."""
        connections = []
        processed_pairs = set()
        
        # Try direct access to connections
        if hasattr(self.system, 'connections') and isinstance(self.system.connections, list):
            return self.system.connections
        elif hasattr(self.system, 'get_connections'):
            return self.system.get_connections()
            
        # If not available, extract from node data
        for node_id, node in nodes_data.items():
            # Handle different node structures
            node_connections = None
            
            if hasattr(node, 'connections') and isinstance(node.connections, dict):
                # Object with connections attribute
                node_connections = node.connections
            elif isinstance(node, dict) and 'connections' in node:
                # Dict with 'connections' key
                node_connections = node['connections']
                
            if node_connections:
                for other_id, strength in node_connections.items():
                    # Create canonical connection ID to avoid duplicates
                    conn_pair = tuple(sorted([node_id, other_id]))
                    if conn_pair in processed_pairs:
                        continue
                        
                    # Only include connection if other node exists
                    if other_id in nodes_data:
                        connections.append({
                            "source": node_id,
                            "target": other_id,
                            "strength": float(strength)
                        })
                        processed_pairs.add(conn_pair)
        
        return connections
    
    def _prepare_visualization_data(self, nodes_data: Dict[str, Any], 
                                  connections_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare visualization data in the format expected by the front-end."""
        # Process nodes for visualization
        nodes_viz = []
        
        # Limit number of nodes for performance
        node_ids = list(nodes_data.keys())
        if len(node_ids) > self.params["max_nodes"]:
            # Keep most important nodes (higher energy or more connections)
            node_importance = {}
            for node_id, node in nodes_data.items():
                # Calculate importance based on energy and connection count
                importance = 0
                
                # Get energy
                if hasattr(node, 'energy'):
                    importance += node.energy
                elif isinstance(node, dict) and 'energy' in node:
                    importance += node['energy']
                
                # Get connection count
                if hasattr(node, 'connections'):
                    importance += len(node.connections) * 0.1
                elif isinstance(node, dict) and 'connections' in node:
                    importance += len(node['connections']) * 0.1
                
                node_importance[node_id] = importance
            
            # Sort by importance and take top N
            node_ids = sorted(node_ids, key=lambda nid: node_importance.get(nid, 0), reverse=True)
            node_ids = node_ids[:self.params["max_nodes"]]
        
        # Process selected nodes
        for node_id in node_ids:
            node = nodes_data[node_id]
            
            # Extract properties based on node structure
            if hasattr(node, 'to_dict'):
                # Object with to_dict method
                node_props = node.to_dict()
            elif isinstance(node, dict):
                # Already a dictionary
                node_props = node
            else:
                # Extract properties directly
                node_props = {
                    "id": node_id,
                    "energy": getattr(node, 'energy', 0.5),
                    "stability": getattr(node, 'stability', 0.8),
                    "features": getattr(node, 'features', []),
                    "metadata": getattr(node, 'metadata', {})
                }
            
            # Get node position or generate one
            position = None
            if 'position' in node_props and node_props['position']:
                position = node_props['position']
                # Ensure 3D position
                if len(position) < 3:
                    position.extend([0] * (3 - len(position)))
                elif len(position) > 3:
                    position = position[:3]
            else:
                # Get features
                features = node_props.get('features', [])
                if not features:
                    # Generate random features
                    features = [random.uniform(-1, 1) for _ in range(self.dimension)]
                
                # Project to 3D
                position = self.projector.project_features(features, node_id)
            
            # Count connections
            connection_count = 0
            if 'connections' in node_props:
                connection_count = len(node_props['connections'])
            
            # Create visualization node
            node_viz = {
                "id": node_id,
                "position": position,
                "energy": float(node_props.get('energy', 0.5)),
                "stability": float(node_props.get('stability', 0.8)),
                "metadata": node_props.get('metadata', {}),
                "numConnections": connection_count
            }
            
            nodes_viz.append(node_viz)
        
        # Filter connections to only include visible nodes and with sufficient strength
        connections_viz = []
        visible_node_ids = set(n["id"] for n in nodes_viz)
        
        for conn in connections_data:
            if (conn['source'] in visible_node_ids and 
                conn['target'] in visible_node_ids and
                conn.get('strength', 0) >= self.params['min_edge_strength']):
                connections_viz.append(conn)
        
        return {
            "nodes": nodes_viz,
            "connections": connections_viz,
            "timestamp": time.time(),
            "params": self.params
        }


def patch_visualization_system(system):
    """
    Patch the Autonomous Kaleidoscope System with visualization capabilities.
    This fixes the AttributeError by adding the missing get_visualization_data method.
    
    Args:
        system: The Autonomous Kaleidoscope System instance to patch
        
    Returns:
        The visualization adapter instance
    """
    # Determine dimension from system if possible
    dimension = 128  # Default
    if hasattr(system, 'dimension'):
        dimension = system.dimension
    elif hasattr(system, 'engine') and hasattr(system.engine, 'dimension'):
        dimension = system.engine.dimension
    
    # Create the visualization adapter
    adapter = VisualizationAdapter(system, dimension=dimension)
    
    # Add the missing method to the system
    system.get_visualization_data = adapter.get_visualization_data
    
    logger.info(f"Patched {system.__class__.__name__} with visualization capabilities")
    return adapter


# When running as standalone script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization_integration.py <path_to_system_module> [--apply]")
        sys.exit(1)
    
    # Parse arguments
    module_path = sys.argv[1]
    apply_patch = "--apply" in sys.argv
    
    if apply_patch:
        print(f"Attempting to patch {module_path}...")
        
        # This would attempt to import the module and patch it
        # Note: This is simplified and would need more error handling in practice
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("system_module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find system instance in module
            system_instance = None
            for name in dir(module):
                obj = getattr(module, name)
                if hasattr(obj, '__class__') and 'System' in obj.__class__.__name__:
                    system_instance = obj
                    break
            
            if system_instance:
                adapter = patch_visualization_system(system_instance)
                print(f"Successfully patched {system_instance.__class__.__name__}")
            else:
                print("Could not find system instance in module")
        except Exception as e:
            print(f"Error patching system: {e}")
    else:
        print("Add the visualization code to your system with:")
        print(f"  from {os.path.basename(__file__)[:-3]} import patch_visualization_system")
        print(f"  patch_visualization_system(your_system_instance)")
