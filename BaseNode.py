import numpy as np
import uuid
from typing import Dict, List, Any, Optional

class Connection:
    def __init__(self, target_id: str, strength: float = 1.0, conn_type: str = "standard"):
        self.target_id = target_id
        self.strength = strength
        self.type = conn_type
        self.stress = 0.0  # Current stress on this connection
        self.age = 0       # Age of connection in simulation steps

class BaseNode:
    def __init__(self, node_id: Optional[str] = None, node_type: str = "standard", 
                 position: Optional[List[float]] = None, attributes: Optional[Dict] = None):
        # Basic properties
        self.id = node_id or str(uuid.uuid4())
        self.type = node_type
        self.position = np.array(position or [0.0, 0.0, 0.0])  # 3D position vector
        self.attributes = attributes or {}
        self.connections = []  # List of Connection objects
        
        # Cube dynamics properties
        self.stress_tensor = np.zeros((3, 3))  # 3x3 stress tensor
        self.energy_level = 1.0                # Current energy level (0.0 to 1.0)
        self.resonance_frequency = np.random.uniform(0.8, 1.2)  # Unique resonance frequency
        self.adaptation_rate = 0.05            # How quickly node adapts to stress
        self.age = 0                           # Age in simulation steps
        
    def add_connection(self, target_id: str, strength: float = 1.0, conn_type: str = "standard") -> Connection:
        """Add a new connection to another node."""
        connection = Connection(target_id, strength, conn_type)
        self.connections.append(connection)
        return connection
    
    def remove_connection(self, target_id: str) -> bool:
        """Remove a connection to another node."""
        for i, conn in enumerate(self.connections):
            if conn.target_id == target_id:
                del self.connections[i]
                return True
        return False
    
    def update_position(self, force_vector: np.ndarray, time_step: float = 0.1):
        """Update the node's position based on forces applied to it."""
        # Simple physics: position += velocity * time
        # where velocity is proportional to force
        self.position += force_vector * time_step
        
        # Update stress tensor based on applied force
        force_magnitude = np.linalg.norm(force_vector)
        if force_magnitude > 0:
            force_direction = force_vector / force_magnitude
            # Outer product to create a stress tensor component
            stress_component = np.outer(force_direction, force_direction) * force_magnitude
            # Update the stress tensor with some decay from previous state
            self.stress_tensor = 0.9 * self.stress_tensor + 0.1 * stress_component
    
    def calculate_stress(self, node_positions: Dict[str, np.ndarray]) -> float:
        """Calculate the total stress on this node based on its connections."""
        total_stress = 0.0
        
        for connection in self.connections:
            if connection.target_id in node_positions:
                target_pos = node_positions[connection.target_id]
                # Calculate distance between nodes
                displacement = target_pos - self.position
                distance = np.linalg.norm(displacement)
                
                # Calculate stress based on ideal distance (could be customized)
                ideal_distance = connection.strength * 2.0  # Example formula
                stress = abs(distance - ideal_distance) / ideal_distance
                
                # Update connection stress
                connection.stress = stress
                total_stress += stress
        
        return total_stress
    
    def adapt_to_stress(self):
        """Adapt node properties based on current stress."""
        # Get the trace of the stress tensor (sum of diagonal elements)
        stress_level = np.trace(self.stress_tensor)
        
        # Adjust energy level based on stress
        if stress_level > 0.8:
            # High stress depletes energy
            self.energy_level = max(0.1, self.energy_level - self.adaptation_rate)
        elif stress_level < 0.2:
            # Low stress allows energy recovery
            self.energy_level = min(1.0, self.energy_level + self.adaptation_rate/2)
        
        # Adapt resonance frequency based on stress pattern
        self.resonance_frequency += (stress_level - 0.5) * 0.01
        self.resonance_frequency = max(0.5, min(1.5, self.resonance_frequency))
        
        # Age the node
        self.age += 1