from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import networkx as nx
import numpy as np
from collections import defaultdict

@dataclass
class Insight:
    """Represents a synthesized insight."""
    insight_id: str
    patterns: List[str]
    relationships: List[tuple]
    confidence: float
    origin_banks: List[str]
    synthesis_depth: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class InsightSynthesizer:
    """Synthesizes new insights from memory bank patterns."""
    
    def __init__(self):
        self.insight_graph = nx.DiGraph()
        self.pattern_relationships = defaultdict(set)
        self.insight_history = []
        self.synthesis_threshold = 0.7
        self.max_synthesis_depth = 3
        
    def synthesize_insights(self, memory_banks: Dict[str, Any]) -> List[Insight]:
        """Generates new insights by analyzing patterns across memory banks."""
        # Extract patterns from memory banks
        active_patterns = self._collect_active_patterns(memory_banks)
        
        # Find pattern relationships
        self._update_pattern_relationships(active_patterns)
        
        # Generate initial insights
        base_insights = self._generate_base_insights(active_patterns)
        
        # Synthesize deeper insights
        deep_insights = self._synthesize_deep_insights(base_insights)
        
        # Combine and filter insights
        all_insights = base_insights + deep_insights
        filtered_insights = self._filter_insights(all_insights)
        
        # Record synthesis results
        self._record_synthesis(filtered_insights)
        
        return filtered_insights
        
    def _collect_active_patterns(self, memory_banks: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Collects active patterns from memory banks."""
        active_patterns = defaultdict(list)
        
        for bank_id, bank in memory_banks.items():
            patterns = bank.get('patterns', [])
            for pattern in patterns:
                if pattern.get('strength', 0) > self.synthesis_threshold:
                    active_patterns[bank_id].append(pattern)
                    
        return active_patterns
        
    def _update_pattern_relationships(self, active_patterns: Dict[str, List[Any]]):
        """Updates relationships between patterns."""
        for bank_id, patterns in active_patterns.items():
            for pattern1 in patterns:
                for pattern2 in patterns:
                    if pattern1['id'] != pattern2['id']:
                        similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                        if similarity > self.synthesis_threshold:
                            self.pattern_relationships[pattern1['id']].add(pattern2['id'])
                            
    def _generate_base_insights(self, active_patterns: Dict[str, List[Any]]) -> List[Insight]:
        """Generates initial insights from related patterns."""
        base_insights = []
        
        # Find strongly connected pattern groups
        for bank_id, patterns in active_patterns.items():
            for pattern in patterns:
                related_patterns = self.pattern_relationships[pattern['id']]
                if related_patterns:
                    insight = self._create_insight(
                        patterns=[pattern['id']] + list(related_patterns),
                        origin_banks=[bank_id],
                        synthesis_depth=1
                    )
                    base_insights.append(insight)
                    
        return base_insights
        
    def _synthesize_deep_insights(self, base_insights: List[Insight]) -> List[Insight]:
        """Synthesizes deeper insights by combining base insights."""
        deep_insights = []
        processed_combinations = set()
        
        for depth in range(2, self.max_synthesis_depth + 1):
            current_insights = []
            
            # Try combining insights at current depth
            for i, insight1 in enumerate(base_insights):
                for insight2 in base_insights[i+1:]:
                    combination_key = tuple(sorted([insight1.insight_id, insight2.insight_id]))
                    
                    if combination_key not in processed_combinations:
                        processed_combinations.add(combination_key)
                        
                        if self._can_combine_insights(insight1, insight2):
                            combined_insight = self._combine_insights(
                                insight1,
                                insight2,
                                depth
                            )
                            if combined_insight:
                                current_insights.append(combined_insight)
                                
            deep_insights.extend(current_insights)
            base_insights = current_insights  # Use current insights for next level
            
        return deep_insights
        
    def _can_combine_insights(self, insight1: Insight, insight2: Insight) -> bool:
        """Determines if two insights can be combined."""
        # Check for pattern overlap
        common_patterns = set(insight1.patterns) & set(insight2.patterns)
        if not common_patterns:
            return False
            
        # Check relationship compatibility
        rel_patterns1 = {p for rel in insight1.relationships for p in rel}
        rel_patterns2 = {p for rel in insight2.relationships for p in rel}
        
        return bool(rel_patterns1 & rel_patterns2)
        
    def _combine_insights(self, insight1: Insight, insight2: Insight, depth: int) -> Optional[Insight]:
        """Combines two insights into a deeper insight."""
        # Combine patterns
        combined_patterns = list(set(insight1.patterns + insight2.patterns))
        
        # Combine relationships
        combined_relationships = list(set(insight1.relationships + insight2.relationships))
        
        # Calculate combined confidence
        combined_confidence = min(insight1.confidence, insight2.confidence) * 0.9
        
        if combined_confidence > self.synthesis_threshold:
            return self._create_insight(
                patterns=combined_patterns,
                relationships=combined_relationships,
                origin_banks=list(set(insight1.origin_banks + insight2.origin_banks)),
                synthesis_depth=depth,
                confidence=combined_confidence
            )
        return None
        
    def _create_insight(
        self,
        patterns: List[str],
        origin_banks: List[str],
        synthesis_depth: int,
        relationships: List[tuple] = None,
        confidence: float = None
    ) -> Insight:
        """Creates a new insight."""
        insight_id = f"insight_{datetime.now().timestamp()}"
        
        if confidence is None:
            confidence = self._calculate_insight_confidence(patterns)
            
        if relationships is None:
            relationships = self._extract_relationships(patterns)
            
        insight = Insight(
            insight_id=insight_id,
            patterns=patterns,
            relationships=relationships,
            confidence=confidence,
            origin_banks=origin_banks,
            synthesis_depth=synthesis_depth,
            metadata={
                'pattern_count': len(patterns),
                'relationship_count': len(relationships),
                'bank_count': len(origin_banks)
            }
        )
        
        self.insight_graph.add_node(
            insight_id,
            data=insight
        )
        
        return insight
        
    def _filter_insights(self, insights: List[Insight]) -> List[Insight]:
        """Filters insights based on quality and novelty."""
        filtered = []
        seen_patterns = set()
        
        for insight in sorted(insights, key=lambda x: x.confidence, reverse=True):
            pattern_key = tuple(sorted(insight.patterns))
            
            if pattern_key not in seen_patterns and insight.confidence > self.synthesis_threshold:
                filtered.append(insight)
                seen_patterns.add(pattern_key)
                
        return filtered
        
    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculates similarity between two patterns."""
        # Type similarity
        type_similarity = 1.0 if pattern1['type'] == pattern2['type'] else 0.0
        
        # Feature similarity
        feature_similarity = self._calculate_feature_similarity(
            pattern1.get('features', {}),
            pattern2.get('features', {})
        )
        
        # Combine similarities
        return 0.4 * type_similarity + 0.6 * feature_similarity
        
    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculates similarity between pattern features."""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
            
        similarities = []
        for feature in common_features:
            if isinstance(features1[feature], (int, float)) and isinstance(features2[feature], (int, float)):
                # Numerical feature
                max_val = max(features1[feature], features2[feature])
                min_val = min(features1[feature], features2[feature])
                if max_val != 0:
                    similarities.append(min_val / max_val)
            else:
                # Non-numerical feature
                similarities.append(1.0 if features1[feature] == features2[feature] else 0.0)
                
        return np.mean(similarities) if similarities else 0.0
        
    def _calculate_insight_confidence(self, patterns: List[str]) -> float:
        """Calculates confidence score for an insight."""
        pattern_relationships = sum(
            1 for p1 in patterns
            for p2 in patterns
            if p2 in self.pattern_relationships[p1]
        )
        
        max_relationships = len(patterns) * (len(patterns) - 1)
        if max_relationships == 0:
            return 0.0
            
        return pattern_relationships / max_relationships
        
    def _extract_relationships(self, patterns: List[str]) -> List[tuple]:
        """Extracts relationships between patterns."""
        relationships = []
        for p1 in patterns:
            for p2 in self.pattern_relationships[p1]:
                if p2 in patterns:
                    relationships.append((p1, p2))
        return relationships
        
    def _record_synthesis(self, insights: List[Insight]):
        """Records synthesis results."""
        synthesis_event = {
            'timestamp': datetime.now().isoformat(),
            'total_insights': len(insights),
            'depth_distribution': defaultdict(int),
            'confidence_stats': {
                'mean': np.mean([i.confidence for i in insights]),
                'std': np.std([i.confidence for i in insights])
            }
        }
        
        for insight in insights:
            synthesis_event['depth_distribution'][insight.synthesis_depth] += 1
            
        self.insight_history.append(synthesis_event)
        
    def get_synthesis_metrics(self) -> Dict[str, Any]:
        """Returns metrics about insight synthesis."""
        if not self.insight_history:
            return {}
            
        recent_events = self.insight_history[-10:]
        
        return {
            'total_syntheses': len(self.insight_history),
            'total_insights_generated': sum(
                event['total_insights'] for event in self.insight_history
            ),
            'average_confidence': np.mean([
                event['confidence_stats']['mean']
                for event in recent_events
            ]),
            'depth_distribution': dict(
                sum(
                    (Counter(event['depth_distribution']) for event in recent_events),
                    Counter()
                )
            ),
            'insight_graph_metrics': {
                'total_nodes': self.insight_graph.number_of_nodes(),
                'total_edges': self.insight_graph.number_of_edges(),
                'average_depth': np.mean([
                    data['data'].synthesis_depth
                    for _, data in self.insight_graph.nodes(data=True)
                ])
            }
        }
