#!/usr/bin/env python3
"""
supernode_processor.py

Advanced processing engine for SuperNode operations.
Implements speculative reasoning, pattern discovery, and topological analysis.
"""

import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
import asyncio
import random
import time
import logging
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import heapq
from gudhi import SimplexTree
from enum import Enum, auto
from collections import defaultdict, deque
import math

from supernode_core import SuperNodeCore, SuperNodeDNA, SuperNodeState, encode_data, decode_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SuperNodeProcessor")

# Define processor constants
MAX_SPECULATION_DEPTH = 5  # Maximum depth for speculative reasoning
MAX_INSIGHT_COUNT = 1000   # Maximum number of insights to store
MAX_PERSPECTIVE_COUNT = 100  # Maximum number of perspectives to store
CORRELATION_THRESHOLD = 0.6  # Threshold for pattern correlation
NOVELTY_THRESHOLD = 0.4  # Threshold for novelty detection
CONFIDENCE_THRESHOLD = 0.7  # Threshold for high-confidence insights

class PatternType(Enum):
    """Types of patterns that can be detected"""
    STRUCTURAL = auto()  # Structural patterns in data topology
    SEQUENTIAL = auto()  # Sequential patterns in time series
    CAUSAL = auto()      # Causal relationships
    HIERARCHICAL = auto() # Hierarchical relationships
    SEMANTIC = auto()    # Semantic/conceptual patterns
    ANOMALY = auto()     # Anomalies and outliers

class InsightType(Enum):
    """Types of insights that can be generated"""
    CORRELATION = auto()  # Statistical correlations
    CAUSATION = auto()    # Causal relationships
    PREDICTION = auto()   # Predictive insights
    EXPLANATION = auto()  # Explanatory insights
    TRANSFORMATION = auto() # Transformational insights
    INTEGRATION = auto()  # Integration of multiple perspectives

@dataclass
class Pattern:
    """Pattern detected in data"""
    id: str
    type: PatternType
    vector: np.ndarray  # Numerical representation
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    
    def similarity(self, other: 'Pattern') -> float:
        """Calculate similarity with another pattern"""
        if self.vector.shape != other.vector.shape:
            # Reshape to enable comparison
            min_dim = min(len(self.vector), len(other.vector))
            vec1 = self.vector[:min_dim]
            vec2 = other.vector[:min_dim]
        else:
            vec1 = self.vector
            vec2 = other.vector
            
        # Normalized dot product similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

@dataclass
class Insight:
    """Insight derived from patterns"""
    id: str
    type: InsightType
    patterns: List[str]  # Pattern IDs
    vector: np.ndarray   # Numerical representation
    description: str
    confidence: float = 0.5
    importance: float = 0.5
    novelty: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Speculation:
    """Speculative extension of patterns and insights"""
    id: str
    source_ids: List[str]  # Source pattern or insight IDs
    vector: np.ndarray
    confidence: float
    plausibility: float
    depth: int = 1  # Speculation depth level
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Perspective:
    """Integrated view combining multiple insights"""
    id: str
    insight_ids: List[str]
    vector: np.ndarray
    strength: float
    coherence: float
    novelty: float
    impact: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class SpeculationEngine:
    """
    Engine for generating speculative extensions of patterns and insights.
    Implements advanced reasoning for exploring pattern possibilities.
    """
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.speculation_graph = nx.DiGraph()
        self.pattern_embeddings = {}  # Pattern ID -> Vector mapping
        self.insight_embeddings = {}  # Insight ID -> Vector mapping
        
    def add_pattern(self, pattern: Pattern) -> None:
        """Add pattern to speculation engine"""
        self.pattern_embeddings[pattern.id] = pattern.vector
        self.speculation_graph.add_node(
            pattern.id,
            type="pattern",
            pattern_type=pattern.type.name,
            confidence=pattern.confidence
        )
        
    def add_insight(self, insight: Insight) -> None:
        """Add insight to speculation engine"""
        self.insight_embeddings[insight.id] = insight.vector
        self.speculation_graph.add_node(
            insight.id,
            type="insight",
            insight_type=insight.type.name,
            confidence=insight.confidence
        )
        
        # Connect insight to its patterns
        for pattern_id in insight.patterns:
            if pattern_id in self.pattern_embeddings:
                self.speculation_graph.add_edge(
                    insight.id, pattern_id,
                    weight=insight.confidence
                )
    
    def generate_speculations(self, source_id: str, depth: int = 1) -> List[Speculation]:
        """
        Generate speculations from a source pattern or insight.
        
        Args:
            source_id: ID of source pattern or insight
            depth: Speculation depth level
            
        Returns:
            List of generated speculations
        """
        if depth > MAX_SPECULATION_DEPTH:
            return []
            
        if source_id not in self.speculation_graph:
            return []
            
        # Get source vector
        if source_id in self.pattern_embeddings:
            source_vector = self.pattern_embeddings[source_id]
            source_type = "pattern"
        elif source_id in self.insight_embeddings:
            source_vector = self.insight_embeddings[source_id]
            source_type = "insight"
        else:
            return []
            
        # Generate different types of speculations
        speculations = []
        
        # Type 1: Extrapolation - project pattern into future/extension
        extrapolation = self._extrapolate_vector(source_vector, source_type)
        spec_id = f"spec_extrap_{source_id}_{depth}_{int(time.time())}"
        extrapolation_confidence = 0.9 / (depth + 1)  # Decreases with depth
        
        speculations.append(Speculation(
            id=spec_id,
            source_ids=[source_id],
            vector=extrapolation,
            confidence=extrapolation_confidence,
            plausibility=self._calculate_plausibility(extrapolation),
            depth=depth,
            metadata={"type": "extrapolation"}
        ))
        
        # Type 2: Counterfactual - invert key dimensions
        counterfactual = self._generate_counterfactual(source_vector)
        spec_id = f"spec_counter_{source_id}_{depth}_{int(time.time())}"
        counter_confidence = 0.7 / (depth + 1)
        
        speculations.append(Speculation(
            id=spec_id,
            source_ids=[source_id],
            vector=counterfactual,
            confidence=counter_confidence,
            plausibility=self._calculate_plausibility(counterfactual),
            depth=depth,
            metadata={"type": "counterfactual"}
        ))
        
        # Type 3: Boundary - explore edge cases
        boundary = self._find_boundary_conditions(source_vector)
        spec_id = f"spec_boundary_{source_id}_{depth}_{int(time.time())}"
        boundary_confidence = 0.5 / (depth + 1)
        
        speculations.append(Speculation(
            id=spec_id,
            source_ids=[source_id],
            vector=boundary,
            confidence=boundary_confidence,
            plausibility=self._calculate_plausibility(boundary),
            depth=depth,
            metadata={"type": "boundary"}
        ))
        
        # Add speculations to graph
        for spec in speculations:
            self.speculation_graph.add_node(
                spec.id,
                type="speculation",
                confidence=spec.confidence,
                plausibility=spec.plausibility,
                depth=depth
            )
            self.speculation_graph.add_edge(
                source_id, spec.id,
                weight=spec.confidence
            )
            
        # Recursively generate deeper speculations with decreasing probability
        if depth < MAX_SPECULATION_DEPTH and random.random() < 1.0 / (depth + 1):
            for spec in speculations:
                deeper_specs = self.generate_speculations(spec.id, depth + 1)
                speculations.extend(deeper_specs)
                
        return speculations
    
    def _extrapolate_vector(self, vector: np.ndarray, source_type: str) -> np.ndarray:
        """Extrapolate vector based on its pattern"""
        # Different extrapolation strategies based on source type
        if source_type == "pattern":
            # For patterns, extend dominant trends
            fft = np.fft.rfft(vector)
            # Get dominant frequencies
            dominant_idx = np.argsort(np.abs(fft))[-5:]  # Top 5 frequencies
            # Amplify dominant frequencies
            boost = np.ones_like(fft)
            boost[dominant_idx] = 1.5
            extrapolated_fft = fft * boost
            # Inverse FFT
            extrapolated = np.fft.irfft(extrapolated_fft, n=len(vector))
            
        else:  # insight
            # For insights, create recombination with noise
            extrapolated = vector + np.random.randn(len(vector)) * 0.1
            # Apply nonlinearity
            extrapolated = np.tanh(extrapolated * 1.2)
            
        # Normalize
        norm = np.linalg.norm(extrapolated)
        if norm > 1e-10:
            extrapolated = extrapolated / norm
            
        return extrapolated
    
    def _generate_counterfactual(self, vector: np.ndarray) -> np.ndarray:
        """Generate counterfactual by flipping important dimensions"""
        # Identify most important dimensions (highest absolute values)
        important_dims = np.argsort(np.abs(vector))[-int(len(vector) * 0.2):]  # Top 20%
        
        # Create counterfactual by flipping sign of important dimensions
        counterfactual = vector.copy()
        counterfactual[important_dims] = -counterfactual[important_dims]
        
        # Add some noise to create variation
        counterfactual += np.random.randn(len(vector)) * 0.05
        
        # Normalize
        norm = np.linalg.norm(counterfactual)
        if norm > 1e-10:
            counterfactual = counterfactual / norm
            
        return counterfactual
    
    def _find_boundary_conditions(self, vector: np.ndarray) -> np.ndarray:
        """Find boundary conditions for the vector"""
        # Create boundary condition by pushing vector to its extremes
        boundary = vector.copy()
        
        # Get dimensions with low but non-zero values
        low_dims = np.where((np.abs(vector) > 0.01) & (np.abs(vector) < 0.3))[0]
        
        if len(low_dims) > 0:
            # Amplify these dimensions to explore boundaries
            boundary[low_dims] *= 3.0
            
        # Apply soft clipping
        boundary = np.tanh(boundary)
        
        # Add directional noise
        noise = np.random.randn(len(vector)) * 0.1
        # Ensure noise pushes in same direction as original vector (where significant)
        sig_dims = np.abs(vector) > 0.3
        noise[sig_dims] = np.abs(noise[sig_dims]) * np.sign(vector[sig_dims])
        
        boundary += noise
        
        # Normalize
        norm = np.linalg.norm(boundary)
        if norm > 1e-10:
            boundary = boundary / norm
            
        return boundary
    
    def _calculate_plausibility(self, vector: np.ndarray) -> float:
        """Calculate plausibility of a vector based on similarity to existing patterns"""
        if not self.pattern_embeddings:
            return 0.5  # Default plausibility
            
        # Calculate similarity to known patterns
        similarities = []
        for pattern_vec in self.pattern_embeddings.values():
            # Ensure compatible shapes
            min_dim = min(len(vector), len(pattern_vec))
            v1 = vector[:min_dim]
            v2 = pattern_vec[:min_dim]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:
                similarities.append(0.0)
            else:
                sim = np.dot(v1, v2) / (norm1 * norm2)
                similarities.append(sim)
                
        # Plausibility is based on similarity to known patterns
        # Higher if somewhat similar but not too similar (novelty balanced with familiarity)
        if not similarities:
            return 0.5
            
        max_sim = max(similarities)
        avg_sim = np.mean(similarities)
        
        # Optimal plausibility at moderate similarity (not too similar, not too different)
        plausibility = 0.5 + 0.5 * (1.0 - np.abs(0.4 - avg_sim) / 0.4) * max_sim
        return min(max(plausibility, 0.1), 0.9)
    
    def find_speculative_connections(self) -> List[Tuple[str, str, float]]:
        """Find potential connections between speculative nodes"""
        # Get all speculative nodes
        spec_nodes = [
            node for node, attrs in self.speculation_graph.nodes(data=True)
            if attrs.get('type') == 'speculation'
        ]
        
        connections = []
        
        # Check all pairs of speculations
        for node1, node2 in itertools.combinations(spec_nodes, 2):
            # Skip if already connected
            if self.speculation_graph.has_edge(node1, node2) or self.speculation_graph.has_edge(node2, node1):
                continue
                
            # Get node vectors (these need to be stored somewhere, perhaps add a dict in init)
            vec1 = None
            vec2 = None
            
            for node, vec in list(self.pattern_embeddings.items()) + list(self.insight_embeddings.items()):
                if node == node1:
                    vec1 = vec
                elif node == node2:
                    vec2 = vec
                    
            if vec1 is None or vec2 is None:
                continue
                
            # Calculate similarity
            min_dim = min(len(vec1), len(vec2))
            similarity = np.dot(vec1[:min_dim], vec2[:min_dim]) / (
                np.linalg.norm(vec1[:min_dim]) * np.linalg.norm(vec2[:min_dim])
            )
            
            # Only connect if similarity is significant
            if similarity > 0.5:
                connections.append((node1, node2, float(similarity)))
                
        return connections
        
    def generate_perspectives(self) -> List[Perspective]:
        """Generate perspectives from speculation graph"""
        # Analyze strongly connected components to identify coherent perspectives
        components = list(nx.strongly_connected_components(self.speculation_graph))
        significant_components = [comp for comp in components if len(comp) >= 3]
        
        perspectives = []
        
        for i, component in enumerate(significant_components):
            # Get insights in this component
            insights = [
                node for node in component
                if self.speculation_graph.nodes[node].get('type') == 'insight'
            ]
            
            if not insights:
                continue
                
            # Compute average vector for the perspective
            vectors = []
            for insight_id in insights:
                if insight_id in self.insight_embeddings:
                    vectors.append(self.insight_embeddings[insight_id])
                    
            if not vectors:
                continue
                
            # Compute component metrics
            avg_vector = np.mean(vectors, axis=0)
            
            # Calculate coherence as average similarity between vectors
            similarities = []
            for vec1, vec2 in itertools.combinations(vectors, 2):
                min_dim = min(len(vec1), len(vec2))
                sim = np.dot(vec1[:min_dim], vec2[:min_dim]) / (
                    np.linalg.norm(vec1[:min_dim]) * np.linalg.norm(vec2[:min_dim])
                )
                similarities.append(sim)
                
            coherence = np.mean(similarities) if similarities else 0.5
            
            # Calculate novelty as 1 - similarity to other components
            other_vectors = []
            for other_comp in significant_components:
                if other_comp == component:
                    continue
                    
                other_insights = [
                    node for node in other_comp
                    if self.speculation_graph.nodes[node].get('type') == 'insight'
                ]
                
                for insight_id in other_insights:
                    if insight_id in self.insight_embeddings:
                        other_vectors.append(self.insight_embeddings[insight_id])
                        
            if other_vectors:
                other_sims = []
                for vec in other_vectors:
                    min_dim = min(len(avg_vector), len(vec))
                    sim = np.dot(avg_vector[:min_dim], vec[:min_dim]) / (
                        np.linalg.norm(avg_vector[:min_dim]) * np.linalg.norm(vec[:min_dim])
                    )
                    other_sims.append(sim)
                    
                avg_other_sim = np.mean(other_sims)
                novelty = 1.0 - avg_other_sim
            else:
                novelty = 0.5
                
            # Impact is proportional to size of component and average node importance
            importance_values = [
                self.speculation_graph.nodes[node].get('confidence', 0.5)
                for node in component
            ]
            avg_importance = np.mean(importance_values) if importance_values else 0.5
            impact = avg_importance * min(1.0, len(component) / 10.0)
            
            # Generate description
            description = f"Perspective integrating {len(insights)} insights"
            
            # Create perspective
            perspective = Perspective(
                id=f"perspective_{i}_{int(time.time())}",
                insight_ids=insights,
                vector=avg_vector,
                strength=avg_importance,
                coherence=coherence,
                novelty=novelty,
                impact=impact,
                description=description,
                metadata={"component_size": len(component)}
            )
            
            perspectives.append(perspective)
            
        return perspectives

class PatternDiscovery:
    """
    Engine for discovering patterns in data.
    Implements algorithms for various types of pattern detection.
    """
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.pattern_library = {}  # Pattern ID -> Pattern mapping
        
    def detect_patterns(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> List[Pattern]:
        """
        Detect patterns in the provided data.
        
        Args:
            data: Input data array
            metadata: Optional metadata about the data
            
        Returns:
            List of detected patterns
        """
        if metadata is None:
            metadata = {}
            
        # Ensure data has the right shape
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        detected_patterns = []
        
        # Apply different pattern detection algorithms based on data characteristics
        
        # 1. Structural pattern detection using SVD
        structural_patterns = self._detect_structural_patterns(data)
        detected_patterns.extend(structural_patterns)
        
        # 2. Sequential pattern detection if data has sequence metadata
        if 'sequence' in metadata or 'time_series' in metadata:
            sequential_patterns = self._detect_sequential_patterns(data)
            detected_patterns.extend(sequential_patterns)
            
        # 3. Detect potential causal patterns if multiple variables present
        if data.shape[0] > 1:
            causal_patterns = self._detect_causal_patterns(data)
            detected_patterns.extend(causal_patterns)
            
        # 4. Hierarchical pattern detection
        hierarchical_patterns = self._detect_hierarchical_patterns(data)
        detected_patterns.extend(hierarchical_patterns)
        
        # 5. Semantic pattern detection
        semantic_patterns = self._detect_semantic_patterns(data, metadata)
        detected_patterns.extend(semantic_patterns)
        
        # 6. Anomaly detection
        anomaly_patterns = self._detect_anomalies(data)
        detected_patterns.extend(anomaly_patterns)
        
        # Store patterns in library
        for pattern in detected_patterns:
            self.pattern_library[pattern.id] = pattern
            
        return detected_patterns
    
    def _detect_structural_patterns(self, data: np.ndarray) -> List[Pattern]:
        """Detect structural patterns using SVD decomposition"""
        patterns = []
        
        # Apply SVD to find principal components
        try:
            U, s, Vh = np.linalg.svd(data, full_matrices=False)
            
            # Keep top components that explain most variance
            explained_variance = s**2 / np.sum(s**2)
            cumulative_variance = np.cumsum(explained_variance)
            
            # Determine significant components (explaining >10% variance)
            significant_idx = np.where(explained_variance > 0.1)[0]
            
            for i in significant_idx:
                component = Vh[i, :]
                
                # Create pattern
                pattern_id = f"struct_pattern_{int(time.time())}_{i}"
                confidence = float(explained_variance[i])
                
                pattern = Pattern(
                    id=pattern_id,
                    type=PatternType.STRUCTURAL,
                    vector=component,
                    confidence=confidence,
                    metadata={
                        "explained_variance": float(explained_variance[i]),
                        "cumulative_variance": float(cumulative_variance[i]),
                        "component_index": int(i)
                    }
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.warning(f"Error in structural pattern detection: {e}")
            
        return patterns
    
    def _detect_sequential_patterns(self, data: np.ndarray) -> List[Pattern]:
        """Detect sequential patterns in time series data"""
        patterns = []
        
        try:
            # Compute autocorrelation to find periodicities
            if data.shape[0] > 1:
                # Multiple time series
                for i in range(min(data.shape[0], 5)):  # Process up to 5 time series
                    series = data[i, :]
                    
                    # Compute autocorrelation
                    autocorr = np.correlate(series, series, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]  # Take only positive lags
                    
                    # Normalize
                    autocorr = autocorr / autocorr[0]
                    
                    # Find peaks in autocorrelation (potential periodicities)
                    peaks = []
                    for j in range(1, len(autocorr)-1):
                        if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1] and autocorr[j] > 0.2:
                            peaks.append((j, autocorr[j]))
                            
                    # Create patterns for significant periodicities
                    for period, strength in peaks:
                        if period < 5 or strength < 0.3:
                            continue  # Skip very short periods or weak correlations
                            
                        # Extract the repeating subsequence
                        subsequence = np.zeros(self.dimension)
                        if period < self.dimension:
                            subsequence[:period] = series[:period]
                            
                        pattern_id = f"seq_pattern_{int(time.time())}_{i}_{period}"
                        confidence = float(strength)
                        
                        pattern = Pattern(
                            id=pattern_id,
                            type=PatternType.SEQUENTIAL,
                            vector=subsequence,
                            confidence=confidence,
                            metadata={
                                "periodicity": int(period),
                                "correlation_strength": float(strength),
                                "series_index": int(i)
                            }
                        )
                        
                        patterns.append(pattern)
            else:
                # Single time series
                series = data[0, :]
                
                # Compute autocorrelation
                autocorr = np.correlate(series, series, mode='full')
                autocorr = autocorr[len(autocorr)//2:]  # Take only positive lags
                
                # Normalize
                autocorr = autocorr / autocorr[0]
                
                # Find peaks in autocorrelation (potential periodicities)
                peaks = []
                for j in range(1, len(autocorr)-1):
                    if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1] and autocorr[j] > 0.2:
                        peaks.append((j, autocorr[j]))
                        
                # Create patterns for significant periodicities
                for period, strength in peaks:
                    if period < 5 or strength < 0.3:
                        continue  # Skip very short periods or weak correlations
                        
                    # Extract the repeating subsequence
                    subsequence = np.zeros(self.dimension)
                    if period < self.dimension:
                        subsequence[:period] = series[:period]
                        
                    pattern_id = f"seq_pattern_{int(time.time())}_{period}"
                    confidence = float(strength)
                    
                    pattern = Pattern(
                        id=pattern_id,
                        type=PatternType.SEQUENTIAL,
                        vector=subsequence,
                        confidence=confidence,
                        metadata={
                            "periodicity": int(period),
                            "correlation_strength": float(strength)
                        }
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.warning(f"Error in sequential pattern detection: {e}")
            
        return patterns
    
    def _detect_causal_patterns(self, data: np.ndarray) -> List[Pattern]:
        """Detect potential causal patterns between variables"""
        patterns = []
        
        try:
            # Simple method: check correlations between variables
            if data.shape[0] < 2:
                return []
                
            # Compute correlation matrix
            corr_matrix = np.corrcoef(data)
            
            # Find variable pairs with high correlation
            high_corr_pairs = []
            for i in range(corr_matrix.shape[0]):
                for j in range(i+1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > 0.7:  # High correlation threshold
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))
                        
            # Create causal patterns for highly correlated pairs
            for i, j, corr in high_corr_pairs:
                # Create a causal pattern vector
                causal_vec = np.zeros(self.dimension)
                
                # Store the pair in vector
                if 2*j+1 < self.dimension:
                    causal_vec[2*i] = 1.0
                    causal_vec[2*i+1] = data[i, :].mean()
                    causal_vec[2*j] = corr
                    causal_vec[2*j+1] = data[j, :].mean()
                    
                pattern_id = f"causal_pattern_{int(time.time())}_{i}_{j}"
                confidence = abs(float(corr))
                
                pattern = Pattern(
                    id=pattern_id,
                    type=PatternType.CAUSAL,
                    vector=causal_vec,
                    confidence=confidence,
                    metadata={
                        "var1_index": int(i),
                        "var2_index": int(j),
                        "correlation": float(corr)
                    }
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.warning(f"Error in causal pattern detection: {e}")
            
        return patterns
    
    def _detect_hierarchical_patterns(self, data: np.ndarray) -> List[Pattern]:
        """Detect hierarchical patterns in data"""
        patterns = []
        
        try:
            # Use a clustering approach to find hierarchical structure
            from scipy.cluster.hierarchy import linkage, fcluster
            
            # Transpose if needed to cluster variables
            if data.shape[0] > data.shape[1]:
                cluster_data = data.T
            else:
                cluster_data = data
                
            # Compute linkage for hierarchical clustering
            Z = linkage(cluster_data, method='ward')
            
            # Cut the dendrogram at different levels to identify hierarchies
            for k in range(2, min(6, cluster_data.shape[0])):
                clusters = fcluster(Z, k, criterion='maxclust')
                
                # Process each cluster
                for cluster_id in range(1, k+1):
                    cluster_members = np.where(clusters == cluster_id)[0]
                    
                    if len(cluster_members) < 2:
                        continue  # Skip singleton clusters
                        
                    # Create hierarchical pattern vector
                    hierarchy_vec = np.zeros(self.dimension)
                    
                    # Store cluster information in vector
                    hierarchy_vec[0] = k  # Number of clusters
                    hierarchy_vec[1] = cluster_id  # This cluster's ID
                    
                    # Store cluster members
                    for i, member in enumerate(cluster_members[:min(20, len(cluster_members))]):
                        idx = 2 + i
                        if idx < self.dimension:
                            hierarchy_vec[idx] = member
                            
                    pattern_id = f"hierarchy_pattern_{int(time.time())}_{k}_{cluster_id}"
                    confidence = 0.5 + 0.1 * len(cluster_members)  # Higher confidence for larger clusters
                    confidence = min(confidence, 0.9)  # Cap at 0.9
                    
                    pattern = Pattern(
                        id=pattern_id,
                        type=PatternType.HIERARCHICAL,
                        vector=hierarchy_vec,
                        confidence=confidence,
                        metadata={
                            "num_clusters": int(k),
                            "cluster_id": int(cluster_id),
                            "cluster_size": int(len(cluster_members)),
                            "cluster_members": [int(m) for m in cluster_members]
                        }
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.warning(f"Error in hierarchical pattern detection: {e}")
            
        return patterns
    
    def _detect_semantic_patterns(self, data: np.ndarray, metadata: Dict[str, Any]) -> List[Pattern]:
        """Detect semantic patterns in data based on metadata"""
        patterns = []
        
        try:
            # Need metadata for semantic interpretation
            if not metadata:
                return []
                
            # Extract semantic features if available
            semantic_keys = [k for k in metadata.keys() if 'semantic' in k or 'concept' in k or 'topic' in k]
            
            if not semantic_keys:
                return []
                
            # Process each semantic feature
            for key in semantic_keys:
                value = metadata[key]
                
                # Convert to string for text-based semantics
                if not isinstance(value, str):
                    value = str(value)
                    
                # Encode semantic content
                semantic_vector = encode_data(value)
                
                # Create pattern
                pattern_id = f"semantic_pattern_{int(time.time())}_{key}"
                confidence = 0.7  # Default confidence for semantic patterns
                
                pattern = Pattern(
                    id=pattern_id,
                    type=PatternType.SEMANTIC,
                    vector=semantic_vector,
                    confidence=confidence,
                    metadata={
                        "semantic_key": key,
                        "semantic_value": value
                    }
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.warning(f"Error in semantic pattern detection: {e}")
            
        return patterns
    
    def _detect_anomalies(self, data: np.ndarray) -> List[Pattern]:
        """Detect anomalies in data"""
        patterns = []
        
        try:
            # Different anomaly detection techniques
            
            # 1. Statistical outlier detection
            if data.shape[0] > 1:
                # Multiple variables
                for i in range(data.shape[0]):
                    series = data[i, :]
                    mean = np.mean(series)
                    std = np.std(series)
                    
                    # Find outliers (> 3 std from mean)
                    outliers = np.where(np.abs(series - mean) > 3 * std)[0]
                    
                    for j, idx in enumerate(outliers):
                        # Create anomaly vector
                        anomaly_vec = np.zeros(self.dimension)
                        anomaly_vec[0] = i  # Variable index
                        anomaly_vec[1] = idx  # Position of anomaly
                        anomaly_vec[2] = series[idx]  # Anomaly value
                        anomaly_vec[3] = (series[idx] - mean) / std  # Z-score
                        
                        pattern_id = f"anomaly_pattern_{int(time.time())}_{i}_{j}"
                        
                        # Confidence based on deviation
                        z_score = abs((series[idx] - mean) / std)
                        confidence = min(0.5 + 0.1 * z_score, 0.95)
                        
                        pattern = Pattern(
                            id=pattern_id,
                            type=PatternType.ANOMALY,
                            vector=anomaly_vec,
                            confidence=confidence,
                            metadata={
                                "variable_index": int(i),
                                "position": int(idx),
                                "value": float(series[idx]),
                                "z_score": float(z_score)
                            }
                        )
                        
                        patterns.append(pattern)
            else:
                # Single variable
                series = data[0, :]
                mean = np.mean(series)
                std = np.std(series)
                
                # Find outliers (> 3 std from mean)
                outliers = np.where(np.abs(series - mean) > 3 * std)[0]
                
                for j, idx in enumerate(outliers):
                    # Create anomaly vector
                    anomaly_vec = np.zeros(self.dimension)
                    anomaly_vec[0] = idx  # Position of anomaly
                    anomaly_vec[1] = series[idx]  # Anomaly value
                    anomaly_vec[2] = (series[idx] - mean) / std  # Z-score
                    
                    pattern_id = f"anomaly_pattern_{int(time.time())}_{j}"
                    
                    # Confidence based on deviation
                    z_score = abs((series[idx] - mean) / std)
                    confidence = min(0.5 + 0.1 * z_score, 0.95)
                    
                    pattern = Pattern(
                        id=pattern_id,
                        type=PatternType.ANOMALY,
                        vector=anomaly_vec,
                        confidence=confidence,
                        metadata={
                            "position": int(idx),
                            "value": float(series[idx]),
                            "z_score": float(z_score)
                        }
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {e}")
            
        return patterns

class InsightGeneration:
    """
    Engine for generating insights from patterns.
    Transforms patterns into higher-level insights and understanding.
    """
    def __init__(self):
        self.insights = {}  # Insight ID -> Insight mapping
        self.patterns = {}  # Pattern ID -> Pattern mapping
        
    def add_pattern(self, pattern: Pattern) -> None:
        """Add pattern to insight generator"""
        self.patterns[pattern.id] = pattern
        
    def generate_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """
        Generate insights from patterns.
        
        Args:
            patterns: List of input patterns
            
        Returns:
            List of generated insights
        """
        # Store patterns
        for pattern in patterns:
            self.patterns[pattern.id] = pattern
            
        # Apply different insight generation strategies
        insights = []
        
        # 1. Correlation insights
        correlation_insights = self._generate_correlation_insights(patterns)
        insights.extend(correlation_insights)
        
        # 2. Causal insights
        causal_insights = self._generate_causal_insights(patterns)
        insights.extend(causal_insights)
        
        # 3. Predictive insights
        predictive_insights = self._generate_predictive_insights(patterns)
        insights.extend(predictive_insights)
        
        # 4. Explanatory insights
        explanatory_insights = self._generate_explanatory_insights(patterns)
        insights.extend(explanatory_insights)
        
        # 5. Transformational insights
        transformational_insights = self._generate_transformational_insights(patterns)
        insights.extend(transformational_insights)
        
        # 6. Integration insights
        integration_insights = self._generate_integration_insights(patterns)
        insights.extend(integration_insights)
        
        # Store generated insights
        for insight in insights:
            self.insights[insight.id] = insight
            
        return insights
    
    def _generate_correlation_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate correlation insights from patterns"""
        insights = []
        
        # Find groups of patterns that are correlated
        pattern_groups = []
        
        # Calculate correlation matrix between patterns
        pattern_vectors = [p.vector for p in patterns]
        n_patterns = len(pattern_vectors)
        
        if n_patterns < 2:
            return []
            
        # Calculate correlation matrix
        corr_matrix = np.zeros((n_patterns, n_patterns))
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                # Calculate correlation
                corr = self._calculate_correlation(pattern_vectors[i], pattern_vectors[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        # Find groups of highly correlated patterns
        for i in range(n_patterns):
            # Find patterns correlated with pattern i
            correlated = [j for j in range(n_patterns) 
                         if i != j and corr_matrix[i, j] > CORRELATION_THRESHOLD]
            
            if correlated:
                group = [i] + correlated
                pattern_groups.append(group)
                
        # Generate insights for each group
        for i, group in enumerate(pattern_groups):
            if len(group) < 2:
                continue
                
            group_patterns = [patterns[idx] for idx in group]
            pattern_ids = [p.id for p in group_patterns]
            
            # Compute average vector
            avg_vector = np.mean([p.vector for p in group_patterns], axis=0)
            
            # Generate description
            description = f"Correlation between {len(group_patterns)} patterns"
            
            # Compute confidence
            confidences = [p.confidence for p in group_patterns]
            avg_confidence = np.mean(confidences)
            
            # Create insight
            insight_id = f"corr_insight_{int(time.time())}_{i}"
            
            insight = Insight(
                id=insight_id,
                type=InsightType.CORRELATION,
                patterns=pattern_ids,
                vector=avg_vector,
                description=description,
                confidence=avg_confidence,
                importance=0.5 + 0.1 * len(group_patterns),  # More patterns -> higher importance
                novelty=0.5,  # Default novelty
                metadata={
                    "pattern_count": len(group_patterns),
                    "average_correlation": float(np.mean([corr_matrix[i, j] for i, j in itertools.combinations(group, 2)]))
                }
            )
            
            insights.append(insight)
            
        return insights
    
    def _generate_causal_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate causal insights from patterns"""
        insights = []
        
        # Look for potential causal relationships
        causal_patterns = [p for p in patterns if p.type == PatternType.CAUSAL]
        
        if not causal_patterns:
            return []
            
        # Process each causal pattern to generate insights
        for i, pattern in enumerate(causal_patterns):
            # Extract useful metadata
            metadata = pattern.metadata
            var1_idx = metadata.get('var1_index', -1)
            var2_idx = metadata.get('var2_index', -1)
            correlation = metadata.get('correlation', 0.0)
            
            if var1_idx < 0 or var2_idx < 0:
                continue
                
            # Generate description
            if correlation > 0:
                relation = "positive correlation"
            else:
                relation = "negative correlation"
                
            description = f"Potential causal relationship ({relation}) between variables {var1_idx} and {var2_idx}"
            
            # Create insight
            insight_id = f"causal_insight_{int(time.time())}_{i}"
            
            insight = Insight(
                id=insight_id,
                type=InsightType.CAUSATION,
                patterns=[pattern.id],
                vector=pattern.vector,
                description=description,
                confidence=pattern.confidence * 0.8,  # Reduce confidence for causal claims
                importance=0.6 + 0.1 * abs(correlation),  # Higher correlation -> higher importance
                novelty=0.6,  # Causal insights often more novel
                metadata={
                    "var1_index": var1_idx,
                    "var2_index": var2_idx,
                    "correlation": correlation,
                    "relation_type": relation
                }
            )
            
            insights.append(insight)
            
        return insights
    
    def _generate_predictive_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate predictive insights from patterns"""
        insights = []
        
        # Look for patterns that enable prediction
        sequential_patterns = [p for p in patterns if p.type == PatternType.SEQUENTIAL]
        
        if not sequential_patterns:
            return []
            
        # Process each sequential pattern to generate predictions
        for i, pattern in enumerate(sequential_patterns):
            # Extract useful metadata
            metadata = pattern.metadata
            periodicity = metadata.get('periodicity', 0)
            strength = metadata.get('correlation_strength', 0.0)
            
            if periodicity < 2:
                continue
                
            # Generate description
            description = f"Predictive pattern with periodicity {periodicity}"
            
            # Create insight
            insight_id = f"predict_insight_{int(time.time())}_{i}"
            
            insight = Insight(
                id=insight_id,
                type=InsightType.PREDICTION,
                patterns=[pattern.id],
                vector=pattern.vector,
                description=description,
                confidence=pattern.confidence,
                importance=0.5 + 0.1 * strength,  # Stronger correlation -> higher importance
                novelty=0.5,  # Default novelty
                metadata={
                    "periodicity": periodicity,
                    "correlation_strength": strength
                }
            )
            
            insights.append(insight)
            
        return insights
    
    def _generate_explanatory_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate explanatory insights from patterns"""
        insights = []
        
        # Use structural and hierarchical patterns for explanation
        structural_patterns = [p for p in patterns if p.type == PatternType.STRUCTURAL]
        hierarchical_patterns = [p for p in patterns if p.type == PatternType.HIERARCHICAL]
        
        if not structural_patterns and not hierarchical_patterns:
            return []
            
        # Combine both types for explanation
        explanatory_patterns = structural_patterns + hierarchical_patterns
        
        # Group related patterns for explanation
        groups = self._group_related_patterns(explanatory_patterns)
        
        # Generate an insight for each group
        for i, group in enumerate(groups):
            if len(group) < 2:
                continue
                
            group_patterns = [p for p in explanatory_patterns if p.id in group]
            pattern_ids = [p.id for p in group_patterns]
            
            # Compute average vector
            avg_vector = np.mean([p.vector for p in group_patterns], axis=0)
            
            # Generate description
            types = [p.type.name for p in group_patterns]
            type_count = {t: types.count(t) for t in set(types)}
            type_str = ", ".join(f"{count} {t}" for t, count in type_count.items())
            
            description = f"Explanatory insight combining {type_str}"
            
            # Compute confidence
            confidences = [p.confidence for p in group_patterns]
            avg_confidence = np.mean(confidences)
            
            # Create insight
            insight_id = f"explain_insight_{int(time.time())}_{i}"
            
            insight = Insight(
                id=insight_id,
                type=InsightType.EXPLANATION,
                patterns=pattern_ids,
                vector=avg_vector,
                description=description,
                confidence=avg_confidence,
                importance=0.5 + 0.05 * len(group_patterns),  # More patterns -> slightly higher importance
                novelty=0.6,  # Explanatory insights often more novel
                metadata={
                    "pattern_types": type_count,
                    "pattern_count": len(group_patterns)
                }
            )
            
            insights.append(insight)
            
        return insights
    
    def _generate_transformational_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate transformational insights from patterns"""
        insights = []
        
        # Need various pattern types for transformation
        if len(patterns) < 3:
            return []
            
        # Look for diverse pattern types
        pattern_types = {p.type for p in patterns}
        
        if len(pattern_types) < 2:
            return []
            
        # Select representative patterns from different types
        representatives = []
        for pattern_type in pattern_types:
            type_patterns = [p for p in patterns if p.type == pattern_type]
            if type_patterns:
                # Choose highest confidence pattern of this type
                best_pattern = max(type_patterns, key=lambda p: p.confidence)
                representatives.append(best_pattern)
                
        if len(representatives) < 2:
            return []
            
        # Create transformational insights by combining patterns
        for i in range(min(3, len(representatives))):
            # Select 2-3 patterns to combine
            combo_size = random.randint(2, min(3, len(representatives)))
            combo = random.sample(representatives, combo_size)
            
            pattern_ids = [p.id for p in combo]
            
            # Compute transformed vector (using SVD to find principal component)
            pattern_matrix = np.vstack([p.vector for p in combo])
            U, s, Vh = np.linalg.svd(pattern_matrix, full_matrices=False)
            transformed_vector = Vh[0, :]  # First principal component
            
            # Generate description
            type_names = [p.type.name for p in combo]
            description = f"Transformational insight combining {', '.join(type_names)}"
            
            # Compute confidence
            confidences = [p.confidence for p in combo]
            avg_confidence = np.mean(confidences)
            
            # Create insight
            insight_id = f"transform_insight_{int(time.time())}_{i}"
            
            insight = Insight(
                id=insight_id,
                type=InsightType.TRANSFORMATION,
                patterns=pattern_ids,
                vector=transformed_vector,
                description=description,
                confidence=avg_confidence * 0.9,  # Slightly reduce confidence for transformations
                importance=0.7,  # Transformational insights often more important
                novelty=0.8,  # Transformational insights are more novel
                metadata={
                    "pattern_types": type_names,
                    "pattern_count": len(combo)
                }
            )
            
            insights.append(insight)
            
        return insights
    
    def _generate_integration_insights(self, patterns: List[Pattern]) -> List[Insight]:
        """Generate integration insights from patterns"""
        insights = []
        
        # Need sufficient patterns for meaningful integration
        if len(patterns) < 4:
            return []
            
        # Classify patterns by confidence
        high_conf = [p for p in patterns if p.confidence > CONFIDENCE_THRESHOLD]
        low_conf = [p for p in patterns if p.confidence <= CONFIDENCE_THRESHOLD]
        
        if not high_conf or not low_conf:
            return []
            
        # Integrate high and low confidence patterns
        # Choose up to 3 high confidence and 2 low confidence patterns
        high_sample = random.sample(high_conf, min(3, len(high_conf)))
        low_sample = random.sample(low_conf, min(2, len(low_conf)))
        
        combo = high_sample + low_sample
        pattern_ids = [p.id for p in combo]
        
        # Generate integrated vector
        # Weighted average, giving more weight to high confidence patterns
        vectors = [p.vector for p in combo]
        weights = [p.confidence for p in combo]
        weights = np.array(weights) / sum(weights)
        
        integrated_vector = np.zeros_like(vectors[0])
        for i, vec in enumerate(vectors):
            integrated_vector += vec * weights[i]
            
        # Generate description
        high_types = [p.type.name for p in high_sample]
        low_types = [p.type.name for p in low_sample]
        
        description = f"Integration of {len(high_sample)} high confidence patterns with {len(low_sample)} exploratory patterns"
        
        # Compute metrics
        avg_confidence = np.mean([p.confidence for p in combo])
        
        # Create insight
        insight_id = f"integration_insight_{int(time.time())}"
        
        insight = Insight(
            id=insight_id,
            type=InsightType.INTEGRATION,
            patterns=pattern_ids,
            vector=integrated_vector,
            description=description,
            confidence=avg_confidence,
            importance=0.6,  # Integration insights are moderately important
            novelty=0.7,  # Integration insights are quite novel
            metadata={
                "high_confidence_types": high_types,
                "low_confidence_types": low_types,
                "total_patterns": len(combo)
            }
        )
        
        insights.append(insight)
        
        return insights
    
    def _calculate_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate correlation between two vectors"""
        if vec1.shape != vec2.shape:
            # Reshape to enable comparison
            min_dim = min(len(vec1), len(vec2))
            v1 = vec1[:min_dim]
            v2 = vec2[:min_dim]
        else:
            v1 = vec1
            v2 = vec2
            
        # Compute correlation
        corr = np.corrcoef(v1, v2)[0, 1]
        
        # Handle NaN
        if np.isnan(corr):
            return 0.0
            
        return float(corr)
    
    def _group_related_patterns(self, patterns: List[Pattern]) -> List[List[str]]:
        """Group related patterns based on similarity"""
        if len(patterns) < 2:
            return [[p.id] for p in patterns]
            
        # Calculate similarity matrix
        n_patterns = len(patterns)
        sim_matrix = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                # Calculate similarity
                sim = patterns[i].similarity(patterns[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
                
        # Use graph-based clustering
        G = nx.Graph()
        
        # Add nodes
        for i, pattern in enumerate(patterns):
            G.add_node(pattern.id, index=i)
            
        # Add edges for similar patterns
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                if sim_matrix[i, j] > CORRELATION_THRESHOLD:
                    G.add_edge(patterns[i].id, patterns[j].id, weight=sim_matrix[i, j])
                    
        # Find connected components (groups)
        groups = list(nx.connected_components(G))
        
        # Convert to list of lists
        return [list(group) for group in groups]

class SuperNodeProcessor:
    """
    Main processor for SuperNode operations.
    Integrates pattern discovery, insight generation, and speculative reasoning.
    """
    def __init__(self, core: SuperNodeCore):
        self.core = core
        self.dimension = core.dimension
        
        # Initialize processing components
        self.pattern_discovery = PatternDiscovery(dimension=self.dimension)
        self.insight_generation = InsightGeneration()
        self.speculation_engine = SpeculationEngine(dimension=self.dimension)
        
        # Storage for processing results
        self.patterns = {}  # ID -> Pattern
        self.insights = {}  # ID -> Insight
        self.speculations = {}  # ID -> Speculation
        self.perspectives = {}  # ID -> Perspective
        
        # Processing parameters
        self.speculation_depth = 3
        self.novelty_threshold = NOVELTY_THRESHOLD
        self.logger = logging.getLogger(f"SuperNodeProcessor_{id(self)}")
    
    def process_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process data through the SuperNode processor.
        
        Args:
            data: Input data array
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary with processing results
        """
        if metadata is None:
            metadata = {}
            
        # Start timer
        start_time = time.time()
        
        # 1. Process data through core
        processed_data = self.core.process_input(data)
        
        # 2. Discover patterns
        patterns = self.pattern_discovery.detect_patterns(processed_data, metadata)
        
        # Store patterns
        for pattern in patterns:
            self.patterns[pattern.id] = pattern
            self.insight_generation.add_pattern(pattern)
            self.speculation_engine.add_pattern(pattern)
        
        # 3. Generate insights
        insights = self.insight_generation.generate_insights(patterns)
        
        # Store insights
        for insight in insights:
            self.insights[insight.id] = insight
            self.speculation_engine.add_insight(insight)
        
        # 4. Generate speculations
        speculations = []
        
        # Choose important insights for speculation
        important_insights = sorted(insights, key=lambda i: i.importance, reverse=True)[:3]
        
        for insight in important_insights:
            spec = self.speculation_engine.generate_speculations(insight.id, depth=self.speculation_depth)
            speculations.extend(spec)
            
        # Store speculations
        for speculation in speculations:
            self.speculations[speculation.id] = speculation
        
        # 5. Find speculative connections
        connections = self.speculation_engine.find_speculative_connections()
        
        # Add connections to speculation graph
        for src, dst, weight in connections:
            self.speculation_engine.speculation_graph.add_edge(src, dst, weight=weight)
        
        # 6. Generate perspectives
        perspectives = self.speculation_engine.generate_perspectives()
        
        # Filter for novel perspectives
        novel_perspectives = [p for p in perspectives if p.novelty > self.novelty_threshold]
        
        # Store perspectives
        for perspective in novel_perspectives:
            self.perspectives[perspective.id] = perspective
        
        # 7. Absorb knowledge into core
        for perspective in novel_perspectives:
            self.core.absorb_knowledge(perspective.vector)
        
        # Compute processing time
        processing_time = time.time() - start_time
        
        # Return results
        return {
            "processed_data": processed_data,
            "pattern_count": len(patterns),
            "insight_count": len(insights),
            "speculation_count": len(speculations),
            "perspective_count": len(novel_perspectives),
            "processing_time": processing_time,
            "patterns": [p.id for p in patterns],
            "insights": [i.id for i in insights],
            "perspectives": [p.id for p in novel_perspectives]
        }
    
    def get_perspective(self, perspective_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a perspective.
        
        Args:
            perspective_id: ID of the perspective
            
        Returns:
            Dictionary with perspective details or None if not found
        """
        if perspective_id not in self.perspectives:
            return None
            
        perspective = self.perspectives[perspective_id]
        
        # Get insight details
        insight_details = []
        for insight_id in perspective.insight_ids:
            if insight_id in self.insights:
                insight = self.insights[insight_id]
                
                # Get pattern details for this insight
                pattern_info = []
                for pattern_id in insight.patterns:
                    if pattern_id in self.patterns:
                        pattern = self.patterns[pattern_id]
                        pattern_info.append({
                            "id": pattern.id,
                            "type": pattern.type.name,
                            "confidence": pattern.confidence,
                            "metadata": pattern.metadata
                        })
                
                insight_details.append({
                    "id": insight.id,
                    "type": insight.type.name,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "importance": insight.importance,
                    "novelty": insight.novelty,
                    "patterns": pattern_info
                })
        
        # Decode perspective vector to text (if possible)
        try:
            decoded_text = decode_data(perspective.vector)
        except:
            decoded_text = "Unable to decode perspective vector to text"
        
        # Return perspective details
        return {
            "id": perspective.id,
            "description": perspective.description,
            "strength": perspective.strength,
            "coherence": perspective.coherence,
            "novelty": perspective.novelty,
            "impact": perspective.impact,
            "insights": insight_details,
            "decoded_text": decoded_text,
            "metadata": perspective.metadata
        }
    
    def get_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an insight.
        
        Args:
            insight_id: ID of the insight
            
        Returns:
            Dictionary with insight details or None if not found
        """
        if insight_id not in self.insights:
            return None
            
        insight = self.insights[insight_id]
        
        # Get pattern details
        pattern_details = []
        for pattern_id in insight.patterns:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern_details.append({
                    "id": pattern.id,
                    "type": pattern.type.name,
                    "confidence": pattern.confidence,
                    "metadata": pattern.metadata
                })
        
        # Decode insight vector to text (if possible)
        try:
            decoded_text = decode_data(insight.vector)
        except:
            decoded_text = "Unable to decode insight vector to text"
        
        # Return insight details
        return {
            "id": insight.id,
            "type": insight.type.name,
            "description": insight.description,
            "confidence": insight.confidence,
            "importance": insight.importance,
            "novelty": insight.novelty,
            "patterns": pattern_details,
            "decoded_text": decoded_text,
            "metadata": insight.metadata
        }
    
    def get_all_insights(self) -> List[Dict[str, Any]]:
        """
        Get summary information about all insights.
        
        Returns:
            List of dictionaries with insight summaries
        """
        return [
            {
                "id": insight.id,
                "type": insight.type.name,
                "description": insight.description,
                "confidence": insight.confidence,
                "importance": insight.importance,
                "novelty": insight.novelty
            }
            for insight in self.insights.values()
        ]
    
    def get_all_perspectives(self) -> List[Dict[str, Any]]:
        """
        Get summary information about all perspectives.
        
        Returns:
            List of dictionaries with perspective summaries
        """
        return [
            {
                "id": perspective.id,
                "description": perspective.description,
                "strength": perspective.strength,
                "coherence": perspective.coherence,
                "novelty": perspective.novelty,
                "impact": perspective.impact
            }
            for perspective in self.perspectives.values()
        ]
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on insights and perspectives.
        
        Returns:
            List of dictionaries with recommendations
        """
        recommendations = []
        
        # Sort perspectives by impact
        top_perspectives = sorted(
            self.perspectives.values(),
            key=lambda p: p.impact,
            reverse=True
        )[:5]  # Top 5 perspectives
        
        # Generate a recommendation for each top perspective
        for i, perspective in enumerate(top_perspectives):
            # Get related insights
            related_insights = [
                self.insights[insight_id]
                for insight_id in perspective.insight_ids
                if insight_id in self.insights
            ]
            
            if not related_insights:
                continue
                
            # Compute average confidence and importance
            avg_confidence = np.mean([insight.confidence for insight in related_insights])
            avg_importance = np.mean([insight.importance for insight in related_insights])
            
            # Generate recommendation ID
            rec_id = f"rec_{i}_{int(time.time())}"
            
            # Extract insight types
            insight_types = [insight.type.name for insight in related_insights]
            
            # Generate recommendation summary
            description = f"Recommendation based on {perspective.description} with {len(related_insights)} insights"
            
            recommendation = {
                "id": rec_id,
                "description": description,
                "perspective_id": perspective.id,
                "confidence": avg_confidence,
                "importance": avg_importance,
                "impact": perspective.impact,
                "insight_types": insight_types,
                "insight_count": len(related_insights)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def merge_with(self, other: 'SuperNodeProcessor') -> 'SuperNodeProcessor':
        """
        Merge this processor with another processor.
        
        Args:
            other: Another SuperNodeProcessor to merge with
            
        Returns:
            New SuperNodeProcessor resulting from the merge
        """
        # Merge the cores
        merged_core = self.core.merge_with(other.core)
        
        # Create new processor with merged core
        merged = SuperNodeProcessor(merged_core)
        
        # Merge patterns
        all_patterns = list(self.patterns.values()) + list(other.patterns.values())
        unique_patterns = {}
        
        for pattern in all_patterns:
            # Check if similar pattern already added
            similar_found = False
            for existing_id, existing in unique_patterns.items():
                similarity = pattern.similarity(existing)
                if similarity > 0.9:  # Very similar patterns
                    # Keep the one with higher confidence
                    if pattern.confidence > existing.confidence:
                        unique_patterns[existing_id] = pattern
                    similar_found = True
                    break
                    
            if not similar_found:
                unique_patterns[pattern.id] = pattern
                
        # Update merged processor patterns
        merged.patterns = unique_patterns
        
        # Rebuild components with merged patterns
        for pattern in unique_patterns.values():
            merged.pattern_discovery.pattern_library[pattern.id] = pattern
            merged.insight_generation.add_pattern(pattern)
            merged.speculation_engine.add_pattern(pattern)
            
        # Merge insights
        all_insights = list(self.insights.values()) + list(other.insights.values())
        unique_insights = {}
        
        for insight in all_insights:
            # Only keep insights with patterns that exist in merged patterns
            valid_patterns = [p for p in insight.patterns if p in unique_patterns]
            if not valid_patterns:
                continue
                
            # Update insight with valid patterns
            insight.patterns = valid_patterns
            unique_insights[insight.id] = insight
            
        # Update merged processor insights
        merged.insights = unique_insights
        
        # Add insights to speculation engine
        for insight in unique_insights.values():
            merged.speculation_engine.add_insight(insight)
            
        # Generate some new perspectives from merged data
        new_perspectives = merged.speculation_engine.generate_perspectives()
        
        # Store new perspectives
        for perspective in new_perspectives:
            merged.perspectives[perspective.id] = perspective
            
        return merged

# Module initialization
if __name__ == "__main__":
    # Basic self-test
    logger.info("SuperNodeProcessor self-test")
    
    # Create a core
    from supernode_core import SuperNodeCore
    core = SuperNodeCore()
    core.start()
    
    # Create processor
    processor = SuperNodeProcessor(core)
    
    # Process some test data
    test_data = np.random.randn(1024)
    result = processor.process_data(test_data)
    
    # Log results
    logger.info(f"Processed {result['pattern_count']} patterns")
    logger.info(f"Generated {result['insight_count']} insights")
    logger.info(f"Created {result['perspective_count']} perspectives")
    
    # Clean up
    core.stop()
    logger.info("SuperNodeProcessor self-test complete")
