import numpy as np
from typing import Dict, List, Optional, Any

class PerspectiveManager:
    """
    Manages multiple perspectives on data and context
    """
    
    def __init__(self):
        self.perspectives = {}
        self.active_perspective = None
    
    def create_perspective(self, name: str, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new perspective with the given name"""
        if name in self.perspectives:
            raise ValueError(f"Perspective {name} already exists")
        
        self.perspectives[name] = initial_data or {}
        return name
    
    def switch_perspective(self, name: str) -> bool:
        """Switch to a different perspective"""
        if name in self.perspectives:
            self.active_perspective = name
            return True
        return False
    
    def get_current_perspective(self) -> Dict[str, Any]:
        """Get the current active perspective"""
        if not self.active_perspective:
            return {}
        return self.perspectives.get(self.active_perspective, {})
    
    def update_perspective(self, name: str, data: Dict[str, Any]) -> bool:
        """Update a perspective with new data"""
        if name in self.perspectives:
            self.perspectives[name].update(data)
            return True
        return False