import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from kaleidoscope_ai.nodes.BaseNode import BaseNode

logger = logging.getLogger(__name__)

class NodeManager:
    def __init__(self, initial_seed: Optional[Dict] = None):
        """Initialize the NodeManager with optional initial seed data."""
        self.nodes = {}  # Dictionary of node_id -> BaseNode
        self.next_node_id = 1
        
        # Initialize with seed if provided
        if initial_seed:
            self._initialize_from_seed(initial_seed)
        else:
            # Create a minimal starting configuration
            self._create_default_seed()
    
    def _initialize_from_seed(self, seed_data: Dict):
        """Initialize nodes from provided seed data."""
        for node_data in seed_data.get("nodes", []):
            node_id = node_data.get("id", f"node_{self.next_node_id}")
            node_type = node_data.get("type", "standard")
            position = node_data.get("position", [0.0, 0.0, 0.0])
            attributes = node_data.get("attributes", {})
            
            node = BaseNode(node_id, node_type, position, attributes)
            self.nodes[node_id] = node
            self.next_node_id += 1
        
        # Create connections after all nodes exist
        for node_data in seed_data.get("nodes", []):
            node_id = node_data.get("id")
            if node_id in self.nodes:
                for conn_data in node_data.get("connections", []):
                    target_id = conn_data.get("target")
                    strength = conn_data.get("strength", 1.0)
                    conn_type = conn_data.get("type", "standard")
                    
                    if target_id in self.nodes:
                        self.nodes[node_id].add_connection(target_id, strength, conn_type)
    
    def _create_default_seed(self):
        """Create a minimal default configuration of nodes."""
        # Create a central node
        central_node = BaseNode(f"node_{self.next_node_id}", "core", [0.0, 0.0, 0.0])
        self.nodes[central_node.id] = central_node
        self.next_node_id += 1
        
        # Create a few surrounding nodes
        for i in range(5):
            angle = (i / 5) * 2 * np.pi
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            z = 0.0
            
            node = BaseNode(f"node_{self.next_node_id}", "standard", [x, y, z])
            self.nodes[node.id] = node
            
            # Connect to central node
            node.add_connection(central_node.id)
            central_node.add_connection(node.id)
            
            self.next_node_id += 1
    
    def create_node(self, node_type: str = "standard", position: Optional[List[float]] = None, 
                   attributes: Optional[Dict] = None) -> BaseNode:
        """Create a new node with given properties."""
        node_id = f"node_{self.next_node_id}"
        node = BaseNode(node_id, node_type, position, attributes)
        self.nodes[node_id] = node
        self.next_node_id += 1
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connections."""
        if node_id in self.nodes:
            # Remove connections to this node from all other nodes
            for other_node in self.nodes.values():
                other_node.remove_connection(node_id)
            
            # Remove the node itself
            del self.nodes[node_id]
            return True
        return False
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> List[BaseNode]:
        """Get all nodes in the system."""
        return list(self.nodes.values())
    
    def get_node_positions(self) -> Dict[str, np.ndarray]:
        """Get a dictionary of all node positions keyed by node ID."""
        return {node_id: node.position for node_id, node in self.nodes.items()}
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in the system."""
        return len(self.nodes)
    
    def update_node_positions(self, time_step: float = 0.1):
        """Update positions of all nodes based on their connections and physics."""
        # Get current positions of all nodes
        node_positions = self.get_node_positions()
        
        # Calculate forces for each node
        for node in self.nodes.values():
            force_vector = self._calculate_force_vector(node, node_positions)
            node.update_position(force_vector, time_step)
            
            # Calculate and adapt to stress
            node.calculate_stress(node_positions)
            node.adapt_to_stress()
    
    def _calculate_force_vector(self, node: BaseNode, node_positions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate the force vector for a node based on its connections."""
        force = np.zeros(3)
        
        for connection in node.connections:
            if connection.target_id in node_positions:
                target_pos = node_positions[connection.target_id]
                # Vector pointing from this node to the target
                direction = target_pos - node.position
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Normalized direction vector
                    normalized_dir = direction / distance
                    
                    # Spring force: F = k * (L - L0)
                    # where k is the spring constant (connection strength)
                    # L is the current length, L0 is the rest length
                    rest_length = 2.0  # Example rest length
                    spring_force = connection.strength * (distance - rest_length)
                    
                    # Apply the force in the appropriate direction
                    force += normalized_dir * spring_force
        
        # Add a small random force for dynamics
        random_force = np.random.uniform(-0.1, 0.1, 3)
        force += random_force
        
        return force
    
    def calculate_total_energy(self) -> float:
        """Calculate the total energy of the network based on node energy levels."""
        return sum(node.energy_level for node in self.nodes.values())
    
    def calculate_network_stress(self) -> float:
        """Calculate the overall stress in the network."""
        if not self.nodes:
            return 0.0
            
        # Average of the trace of all stress tensors
        total_stress = 0.0
        for node in self.nodes.values():
            total_stress += np.trace(node.stress_tensor)
        
        return total_stress / len(self.nodes)