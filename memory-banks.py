from dataclasses import dataclass, field
from typing import Dict, List, Any
import networkx as nx
import numpy as np
from datetime import datetime
import uuid

@dataclass
class MemoryBank:
    """Represents a single memory bank (gear) in the Kaleidoscope Engine."""
    bank_id: str
    capacity: int
    weight: float = 1.0
    insights: List[Dict] = field(default_factory=list)

class KaleidoscopeEngine:
    def __init__(self, num_banks: int = 20, bank_capacity: int = 10):
        self.memory_graph = nx.DiGraph()
        self.num_banks = num_banks
        self.bank_capacity = bank_capacity
        self.current_bank_index = 0
        self._initialize_memory_banks()
    
    def _initialize_memory_banks(self):
        """Initializes the memory bank network structure."""
        # Create memory bank nodes
        for i in range(self.num_banks):
            bank = MemoryBank(
                bank_id=f"bank_{i}",
                capacity=self.bank_capacity,
                weight=np.random.uniform(0.5, 1.5)
            )
            self.memory_graph.add_node(bank.bank_id, bank=bank)
        
        # Create connections between banks
        for i in range(self.num_banks - 1):
            self.memory_graph.add_edge(
                f"bank_{i}", 
                f"bank_{i+1}", 
                weight=np.random.uniform(0.5, 1.5)
            )
    
    def add_insight(self, data: StandardizedData):
        """Adds standardized data to the current memory bank and updates weights."""
        current_bank_id = f"bank_{self.current_bank_index}"
        bank = self.memory_graph.nodes[current_bank_id]["bank"]
        
        if len(bank.insights) >= bank.capacity:
            self._shift_memory_banks()
            bank = self.memory_graph.nodes[current_bank_id]["bank"]
        
        insight = {
            "id": str(uuid.uuid4()),
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "weight": bank.weight
        }
        
        bank.insights.append(insight)
        self._update_weights(current_bank_id, data)
    
    def _shift_memory_banks(self):
        """Shifts insights between memory banks based on weights and connections."""
        sorted_banks = sorted(
            self.memory_graph.nodes(data=True),
            key=lambda x: x[1]["bank"].weight,
            reverse=True
        )
        
        for bank_id, data in sorted_banks:
            bank = data["bank"]
            if len(bank.insights) > bank.capacity:
                overflow = bank.insights[bank.capacity:]
                bank.insights = bank.insights[:bank.capacity]
                
                for successor in self.memory_graph.successors(bank_id):
                    next_bank = self.memory_graph.nodes[successor]["bank"]
                    if len(next_bank.insights) < next_bank.capacity:
                        next_bank.insights.extend(overflow)
                        break
        
        self.current_bank_index = (self.current_bank_index + 1) % self.num_banks
    
    def _update_weights(self, bank_id: str, data: StandardizedData):
        """Updates the weights of memory banks and connections based on new data."""
        bank = self.memory_graph.nodes[bank_id]["bank"]
        bank.weight *= (0.8 + 0.2 * data.confidence)
        
        for source, target in data.relationships:
            if self.memory_graph.has_edge(source, target):
                current_weight = self.memory_graph[source][target]["weight"]
                self.memory_graph[source][target]["weight"] = (
                    current_weight * 0.8 + 0.2 * data.confidence
                )

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the KaleidoscopeEngine."""
        return {
            "current_bank": self.current_bank_index,
            "total_insights": sum(
                len(data["bank"].insights)
                for _, data in self.memory_graph.nodes(data=True)
            )
        }
