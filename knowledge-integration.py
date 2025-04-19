from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
import numpy as np
from datetime import datetime
import networkx as nx
from collections import defaultdict

@dataclass
class KnowledgePattern:
    """Represents a pattern in the knowledge network."""
    pattern_id: str
    elements: List[Any]
    relationships: List[tuple]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class KnowledgeIntegrator:
    """Integrates new knowledge and identifies patterns across the network."""
    
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.pattern_registry = defaultdict(list)
        self.similarity_cache = {}
        self.confidence_threshold = 0.7
        
        # Initialize pattern matchers
        self.matchers = {
            'structural': self._match_structural_patterns,
            'semantic': self._match_semantic_patterns,
            'temporal': self._match_temporal_patterns,
            'causal': self._match_causal_patterns
        }

    def integrate_knowledge(self, new_data: StandardizedData) -> Dict[str, Any]:
        """Integrates new knowledge into the system."""
        # Process and add new data
        node_id = self._add_knowledge_node(new_data)
        
        # Identify and update relationships
        self._update_relationships(node_id, new_data)
        
        # Find patterns
        patterns = self._find_patterns(node_id)
        
        # Generate integration report
        report = self._generate_integration_report(node_id, patterns)
        
        return report

    def _add_knowledge_node(self, data: StandardizedData) -> str:
        """Adds a new knowledge node to the graph."""
        node_id = f"node_{datetime.now().timestamp()}"
        
        self.knowledge_graph.add_node(
            node_id,
            data=data.raw_data,
            metadata=data.metadata,
            type=data.data_type,
            confidence=data.confidence,
            timestamp=data.timestamp
        )
        
        return node_id

    def _update_relationships(self, node_id: str, data: StandardizedData):
        """Updates relationships for the new knowledge node."""
        for existing_node in self.knowledge_graph.nodes():
            if existing_node != node_id:
                similarity = self._calculate_similarity(
                    data,
                    self.knowledge_graph.nodes[existing_node]['data']
                )
                
                if similarity >= self.confidence_threshold:
                    self.knowledge_graph.add_edge(
                        node_id,
                        existing_node,
                        weight=similarity,
                        type='similarity'
                    )

    def _find_patterns(self, node_id: str) -> List[KnowledgePattern]:
        """Identifies patterns related to the new node."""
        patterns = []
        
        # Apply each pattern matcher
        for matcher_type, matcher_func in self.matchers.items():
            matched_patterns = matcher_func(node_id)
            patterns.extend(matched_patterns)
        
        # Register new patterns
        for pattern in patterns:
            self.pattern_registry[pattern.pattern_id].append(pattern)
        
        return patterns

    def _match_structural_patterns(self, node_id: str) -> List[KnowledgePattern]:
        """Identifies structural patterns in the knowledge graph."""
        patterns = []
        
        # Get local subgraph
        local_graph = nx.ego_graph(self.knowledge_graph, node_id, radius=2)
        
        # Find strongly connected components
        components = list(nx.strongly_connected_components(local_graph))
        
        for component in components:
            if len(component) >= 3:  # Minimum size for a pattern
                confidence = self._calculate_component_confidence(component)
                
                if confidence >= self.confidence_threshold:
                    pattern = KnowledgePattern(
                        pattern_id=f"struct_{datetime.now().timestamp()}",
                        elements=list(component),
                        relationships=list(local_graph.subgraph(component).edges()),
                        confidence=confidence,
                        metadata={
                            'type': 'structural',
                            'size': len(component),
                            'density': nx.density(local_graph.subgraph(component))
                        }
                    )
                    patterns.append(pattern)
        
        return patterns

    def _match_semantic_patterns(self, node_id: str) -> List[KnowledgePattern]:
        """Identifies semantic patterns in the knowledge graph."""
        patterns = []
        node_data = self.knowledge_graph.nodes[node_id]['data']
        
        # Find semantically similar nodes
        similar_nodes = []
        for other_node in self.knowledge_graph.nodes():
            if other_node != node_id:
                other_data = self.knowledge_graph.nodes[other_node]['data']
                similarity = self._calculate_semantic_similarity(node_data, other_data)
                
                if similarity >= self.confidence_threshold:
                    similar_nodes.append((other_node, similarity))
        
        # Group similar nodes into patterns
        if similar_nodes:
            pattern = KnowledgePattern(
                pattern_id=f"sem_{datetime.now().timestamp()}",
                elements=[node_id] + [node for node, _ in similar_nodes],
                relationships=[(node_id, node) for node, _ in similar_nodes],
                confidence=np.mean([sim for _, sim in similar_nodes]),
                metadata={
                    'type': 'semantic',
                    'similarity_scores': dict(similar_nodes)
                }
            )
            patterns.append(pattern)
        
        return patterns

    def _match_temporal_patterns(self, node_id: str) -> List[KnowledgePattern]:
        """Identifies temporal patterns in the knowledge graph."""
        patterns = []
        
        # Get temporal sequence
        sequence = self._get_temporal_sequence(node_id)
        
        if len(sequence) >= 3:  # Minimum sequence length
            confidence = self._calculate_sequence_confidence(sequence)
            
            if confidence >= self.confidence_threshold:
                pattern = KnowledgePattern(
                    pattern_id=f"temp_{datetime.now().timestamp()}",
                    elements=sequence,
                    relationships=list(zip(sequence[:-1], sequence[1:])),
                    confidence=confidence,
                    metadata={
                        'type': 'temporal',
                        'sequence_length': len(sequence),
                        'time_span': self._calculate_time_span(sequence)
                    }
                )
                patterns.append(pattern)
        
        return patterns

    def _match_causal_patterns(self, node_id: str) -> List[KnowledgePattern]:
        """Identifies causal patterns in the knowledge graph."""
        patterns = []
        
        # Find potential cause-effect relationships
        causal_chains = self._find_causal_chains(node_id)
        
        for chain in causal_chains:
            confidence = self._calculate_causal_confidence(chain)
            
            if confidence >= self.confidence_threshold:
                pattern = KnowledgePattern(
                    pattern_id=f"causal_{datetime.now().timestamp()}",
                    elements=chain,
                    relationships=list(zip(chain[:-1], chain[1:])),
                    confidence=confidence,
                    metadata={
                        'type': 'causal',
                        'chain_length': len(chain),
                        'causal_strength': self._calculate_causal_strength(chain)
                    }
                )
                patterns.append(pattern)
        
        return patterns

    def _calculate_similarity(self, data1: Any, data2: Any) -> float:
        """Calculates similarity between two pieces of data."""
        cache_key = (str(data1), str(data2))
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Calculate based on data type
        if isinstance(data1, str) and isinstance(data2, str):
            similarity = self._calculate_text_similarity(data1, data2)
        elif isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
            similarity = self._calculate_numerical_similarity(data1, data2)
        else:
            similarity = self._calculate_structural_similarity(data1, data2)
        
        self.similarity_cache[cache_key] = similarity
        return similarity

    def _get_temporal_sequence(self, node_id: str) -> List[str]:
        """Gets temporal sequence related to a node."""
        # Get all connected nodes
        connected = list(nx.descendants(self.knowledge_graph, node_id))
        connected.append(node_id)
        
        # Sort by timestamp
        sequence = sorted(
            connected,
            key=lambda x: self.knowledge_graph.nodes[x]['timestamp']
        )
        
        return sequence

    def _find_causal_chains(self, node_id: str) -> List[List[str]]:
        """Finds potential causal chains involving the node."""
        chains = []
        
        # Get all paths of length 2-4 starting from node
        for length in range(2, 5):
            for path in nx.all_simple_paths(
                self.knowledge_graph,
                node_id,
                None,
                length
            ):
                if self._is_causal_chain(path):
                    chains.append(path)
        
        return chains

    def _is_causal_chain(self, path: List[str]) -> bool:
        """Determines if a path represents a causal chain."""
        # Check temporal ordering
        times = [
            self.knowledge_graph.nodes[node]['timestamp']
            for node in path
        ]
        if not all(t1 < t2 for t1, t2 in zip(times[:-1], times[1:])):
            return False
        
        # Check edge types and weights
        for n1, n2 in zip(path[:-1], path[1:]):
            edge_data = self.knowledge_graph.edges[n1, n2]
            if edge_data.get('type') != 'causal' or edge_data.get('weight', 0) < 0.5:
                return False
        
        return True

    def _calculate_component_confidence(self, component: Set[str]) -> float:
        """Calculates confidence score for a component."""
        if not component:
            return 0.0
        
        subgraph = self.knowledge_graph.subgraph(component)
        
        # Consider multiple factors
        density = nx.density(subgraph)
        avg_confidence = np.mean([
            self.knowledge_graph.nodes[node]['confidence']
            for node in component
        ])
        edge_weights = [
            data.get('weight', 0.5)
            for _, _, data in subgraph.edges(data=True)
        ]
        avg_weight = np.mean(edge_weights) if edge_weights else 0.5
        
        return (density + avg_confidence + avg_weight) / 3

    def _calculate_sequence_confidence(self, sequence: List[str]) -> float:
        """Calculates confidence score for a temporal sequence."""
        if len(sequence) < 2:
            return 0.0
        
        # Consider temporal consistency and relationship strengths
        temporal_scores = []
        relationship_scores = []
        
        for i in range(len(sequence) - 1):
            # Temporal scoring
            time1 = self.knowledge_graph.nodes[sequence[i]]['timestamp']
            time2 = self.knowledge_graph.nodes[sequence[i + 1]]['timestamp']
            temporal_scores.append(1.0 if time1 < time2 else 0.0)
            
            # Relationship scoring
            if self.knowledge_graph.has_edge(sequence[i], sequence[i + 1]):
                weight = self.knowledge_graph.edges[sequence[i], sequence[i + 1]].get('weight', 0.5)
                relationship_scores.append(weight)
            else:
                relationship_scores.append(0.0)
        
        temporal_confidence = np.mean(temporal_scores)
        relationship_confidence = np.mean(relationship_scores)
        
        return (temporal_confidence + relationship_confidence) / 2

    def _calculate_causal_confidence(self, chain: List[str]) -> float:
        """Calculates confidence score for a causal chain."""
        if len(chain) < 2:
            return 0.0
        
        # Consider multiple factors
        temporal_validity = self._check_temporal_validity(chain)
        relationship_strength = self._calculate_relationship_strength(chain)
        supporting_evidence = self._calculate_supporting_evidence(chain)
        
        return (temporal_validity + relationship_strength + supporting_evidence) / 3

    def _generate_integration_report(self, node_id: str, patterns: List[KnowledgePattern]) -> Dict[str, Any]:
        """Generates a report about the knowledge integration."""
        return {
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'patterns_found': len(patterns),
            'pattern_types': {
                pattern_type: len([
                    p for p in patterns
                    if p.metadata['type'] == pattern_type
                ])
                for pattern_type in self.matchers.keys()
            },
            'average_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0.0,
            'patterns': [
                {
                    'id': p.pattern_id,
                    'type': p.metadata['type'],
                    'confidence': p.confidence,
                    'size': len(p.elements)
                }
                for p in patterns
            ]
        }
