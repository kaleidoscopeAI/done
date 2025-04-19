from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
import numpy as np
from datetime import datetime
import uuid

@dataclass
class DomainNode:
    """Represents a specialized node containing domain-specific knowledge."""
    node_id: str
    domain: str
    knowledge: StandardizedData
    connections: Set[str] = field(default_factory=set)
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0

class ClusterManager:
    """Manages the creation and organization of domain-specific node clusters."""
    def __init__(self):
        self.cluster_graph = nx.Graph()
        self.domain_nodes: Dict[str, DomainNode] = {}
        self.clusters: Dict[str, Set[str]] = {}
        self.similarity_threshold = 0.7

    def create_domain_node(self, insight: Dict[str, Any]) -> DomainNode:
        """Creates a new domain node from a generated insight."""
        node_id = str(uuid.uuid4())
        domain = self._extract_domain(insight['data'])
        
        domain_node = DomainNode(
            node_id=node_id,
            domain=domain,
            knowledge=insight['data'],
            confidence=insight['data'].confidence
        )
        
        self.domain_nodes[node_id] = domain_node
        self.cluster_graph.add_node(
            node_id,
            domain=domain,
            confidence=domain_node.confidence
        )
        
        return domain_node

    def _extract_domain(self, data: StandardizedData) -> str:
        """Extracts the domain classification from standardized data."""
        domain_indicators = {
            "cell": ["membrane", "phospholipids", "protein", "nucleus"],
            "chemistry": ["compound", "reaction", "molecule", "bond"],
            "physics": ["force", "energy", "mass", "velocity"]
        }

        text_content = ""
        if data.data_type == "text":
            text_content = data.raw_data
        elif data.data_type == "structured":
            text_content = str(data.metadata)

        # Count domain-specific terms
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_content.lower())
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"

    def update_clusters(self):
        """Updates cluster assignments based on node relationships and domain knowledge."""
        # Reset clusters
        self.clusters = {}
        
        # Find connected components in the cluster graph
        components = nx.connected_components(self.cluster_graph)
        
        for idx, component in enumerate(components):
            # Check if nodes in component belong to the same domain
            domains = {self.domain_nodes[node_id].domain for node_id in component}
            
            if len(domains) == 1:
                # Single domain cluster
                domain = domains.pop()
                cluster_id = f"{domain}_cluster_{idx}"
                self.clusters[cluster_id] = set(component)
            else:
                # Mixed domain cluster - split if necessary
                self._handle_mixed_domain_cluster(component, idx)

    def _handle_mixed_domain_cluster(self, component: Set[str], idx: int):
        """Handles clusters containing nodes from multiple domains."""
        # Group nodes by domain
        domain_groups = {}
        for node_id in component:
            domain = self.domain_nodes[node_id].domain
            if domain not in domain_groups:
                domain_groups[domain] = set()
            domain_groups[domain].add(node_id)

        # Create separate clusters for each domain
        for domain, nodes in domain_groups.items():
            cluster_id = f"{domain}_cluster_{idx}_{len(self.clusters)}"
            self.clusters[cluster_id] = nodes

    def add_connection(self, node1_id: str, node2_id: str, weight: float = 1.0):
        """Adds a weighted connection between two nodes."""
        if node1_id in self.domain_nodes and node2_id in self.domain_nodes:
            self.cluster_graph.add_edge(node1_id, node2_id, weight=weight)
            self.domain_nodes[node1_id].connections.add(node2_id)
            self.domain_nodes[node2_id].connections.add(node1_id)

    def calculate_node_similarity(self, node1_id: str, node2_id: str) -> float:
        """Calculates similarity between two nodes based on their knowledge and domain."""
        node1 = self.domain_nodes[node1_id]
        node2 = self.domain_nodes[node2_id]

        # Domain similarity
        domain_similarity = 1.0 if node1.domain == node2.domain else 0.0

        # Knowledge similarity based on metadata
        knowledge_similarity = self._calculate_knowledge_similarity(
            node1.knowledge,
            node2.knowledge
        )

        # Weighted combination
        return 0.6 * domain_similarity + 0.4 * knowledge_similarity

    def _calculate_knowledge_similarity(
        self,
        knowledge1: StandardizedData,
        knowledge2: StandardizedData
    ) -> float:
        """Calculates similarity between two knowledge representations."""
        # Compare metadata
        common_keys = set(knowledge1.metadata.keys()) & set(knowledge2.metadata.keys())
        if not common_keys:
            return 0.0

        metadata_similarity = sum(
            1 for key in common_keys
            if knowledge1.metadata[key] == knowledge2.metadata[key]
        ) / len(common_keys)

        # Compare relationships
        relationship_similarity = len(
            set(knowledge1.relationships) & set(knowledge2.relationships)
        ) / max(
            len(knowledge1.relationships) + len(knowledge2.relationships),
            1
        )

        return 0.7 * metadata_similarity + 0.3 * relationship_similarity

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Generates a summary of current cluster state."""
        summary = {
            "total_nodes": len(self.domain_nodes),
            "total_clusters": len(self.clusters),
            "clusters": {}
        }

        for cluster_id, nodes in self.clusters.items():
            cluster_info = {
                "size": len(nodes),
                "domain": cluster_id.split("_")[0],
                "average_confidence": np.mean([
                    self.domain_nodes[node_id].confidence
                    for node_id in nodes
                ]),
                "nodes": [
                    {
                        "id": node_id,
                        "connections": len(self.domain_nodes[node_id].connections)
                    }
                    for node_id in nodes
                ]
            }
            summary["clusters"][cluster_id] = cluster_info

        return summary
