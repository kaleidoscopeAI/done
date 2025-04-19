from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
import numpy as np
from datetime import datetime
import networkx as nx

@dataclass
class NetworkState:
    """Represents the current state of the learning network."""
    energy_level: float
    node_density: float
    connection_strength: float
    learning_rate: float
    stability: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class DynamicNetworkManager:
    """Manages network evolution and learning dynamics."""
    def __init__(self, initial_energy: float = 100.0):
        self.graph = nx.DiGraph()
        self.state = NetworkState(
            energy_level=initial_energy,
            node_density=0.0,
            connection_strength=1.0,
            learning_rate=0.1
        )
        self.energy_threshold = 0.3
        self.stability_threshold = 0.7
        self.learning_history = []

    def update_network(self, insights: List[Dict[str, Any]]):
        """Updates network based on new insights."""
        self._consume_energy()
        
        for insight in insights:
            if self.state.energy_level > self.energy_threshold:
                self._process_insight(insight)
                self._update_connections()
                self._adjust_learning_rate()
            else:
                break

        self._record_state()

    def _process_insight(self, insight: Dict[str, Any]):
        """Processes a single insight and updates network structure."""
        # Add new node for insight
        node_id = insight['id']
        self.graph.add_node(
            node_id,
            data=insight['data'],
            weight=insight.get('confidence', 1.0)
        )

        # Create connections with existing nodes
        for existing_node in self.graph.nodes():
            if existing_node != node_id:
                similarity = self._calculate_similarity(
                    insight['data'],
                    self.graph.nodes[existing_node]['data']
                )
                if similarity > self.stability_threshold:
                    self.graph.add_edge(
                        node_id,
                        existing_node,
                        weight=similarity
                    )

    def _calculate_similarity(self, data1: StandardizedData, data2: StandardizedData) -> float:
        """Calculates similarity between two data points."""
        metadata_similarity = self._compare_metadata(data1.metadata, data2.metadata)
        relationship_similarity = self._compare_relationships(
            data1.relationships,
            data2.relationships
        )
        
        return 0.7 * metadata_similarity + 0.3 * relationship_similarity

    def _compare_metadata(self, metadata1: Dict, metadata2: Dict) -> float:
        """Compares metadata dictionaries."""
        common_keys = set(metadata1.keys()) & set(metadata2.keys())
        if not common_keys:
            return 0.0
            
        similarity = sum(
            1 for key in common_keys
            if metadata1[key] == metadata2[key]
        ) / len(common_keys)
        
        return similarity

    def _compare_relationships(self, rel1: List[Tuple], rel2: List[Tuple]) -> float:
        """Compares relationship lists."""
        if not rel1 or not rel2:
            return 0.0
            
        common_relationships = set(rel1) & set(rel2)
        total_relationships = set(rel1) | set(rel2)
        
        return len(common_relationships) / len(total_relationships)

    def _update_connections(self):
        """Updates connection strengths based on network activity."""
        if not self.graph.edges():
            return

        # Calculate average edge weight
        avg_weight = np.mean([
            data['weight'] 
            for _, _, data in self.graph.edges(data=True)
        ])

        # Update connection strengths
        for u, v, data in self.graph.edges(data=True):
            current_weight = data['weight']
            if current_weight > avg_weight:
                # Strengthen strong connections
                new_weight = current_weight * (1 + self.state.learning_rate)
            else:
                # Weaken weak connections
                new_weight = current_weight * (1 - self.state.learning_rate)
            
            self.graph[u][v]['weight'] = np.clip(new_weight, 0.1, 1.0)

    def _adjust_learning_rate(self):
        """Adjusts learning rate based on network stability."""
        if not self.graph.edges():
            return

        # Calculate weight variance
        weights = [
            data['weight'] 
            for _, _, data in self.graph.edges(data=True)
        ]
        weight_variance = np.var(weights)

        # Adjust learning rate inversely to variance
        self.state.learning_rate = np.clip(
            0.1 / (1 + weight_variance),
            0.01,
            0.1
        )

    def _consume_energy(self):
        """Consumes energy based on network activity."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        energy_cost = (
            0.1 * num_nodes +  # Base node maintenance cost
            0.05 * num_edges  # Connection maintenance cost
        )
        
        self.state.energy_level = max(0, self.state.energy_level - energy_cost)

    def _record_state(self):
        """Records current network state."""
        self.learning_history.append({
            'timestamp': datetime.now(),
            'energy_level': self.state.energy_level,
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'learning_rate': self.state.learning_rate,
            'stability': self.state.stability
        })

    def get_network_metrics(self) -> Dict[str, Any]:
        """Returns current network metrics."""
        return {
            'network_size': self.graph.number_of_nodes(),
            'connection_density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'average_path_length': nx.average_shortest_path_length(self.graph)
            if nx.is_connected(self.graph.to_undirected()) else float('inf'),
            'energy_level': self.state.energy_level,
            'learning_rate': self.state.learning_rate,
            'stability': self.state.stability
        }

    def get_learning_history(self) -> List[Dict[str, Any]]:
        """Returns the learning history."""
        return self.learning_history

    def get_strongest_connections(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Returns the strongest connections in the network."""
        return [
            (u, v, data['weight'])
            for u, v, data in self.graph.edges(data=True)
            if data['weight'] >= threshold
        ]

