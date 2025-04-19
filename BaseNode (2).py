import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import uuid

class BaseNode:
    """
    Base class for all nodes in the system
    """
    
    def __init__(self, node_id: Optional[str] = None, node_type: str = "base"):
        self.id = node_id or str(uuid.uuid4())
        self.type = node_type
        self.connections = {}
        self.state = {"energy": 100.0}
        self.created_at = None
        self.updated_at = None
        
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data and update node state
        
        This method should be overridden by subclasses
        """
        return {"status": "processed", "result": input_data}
    
    def connect_to(self, other_node, connection_type: str = "default") -> bool:
        """Connect this node to another node"""
        if other_node.id not in self.connections:
            self.connections[other_node.id] = {
                "node": other_node,
                "type": connection_type
            }
            return True
        return False
    
    def disconnect_from(self, node_id: str) -> bool:
        """Disconnect this node from another node"""
        if node_id in self.connections:
            del self.connections[node_id]
            return True
        return False
    
    def update_state(self, state_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update the node's state with new data"""
        self.state.update(state_updates)
        return self.state
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node"""
        return self.state.copy()
    
    def _record_processing_event(self, result: Dict[str, Any]) -> None:
        """Record processing event for logging/analytics"""
        pass