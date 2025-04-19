# modules/PatternRecognition.py
import logging
import time
import random
import re
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class PatternRecognition:
    """
    Recognizes patterns in data across the system.
    This module analyzes outputs from different nodes and perspectives 
    to identify meaningful patterns.
    """
    
    def __init__(self):
        self.known_patterns = {}  # Pattern storage
        self.pattern_history = []  # Track pattern recognition history
        self.confidence_threshold = 0.7  # Minimum confidence to report a pattern
        
        # Define pattern detectors
        self.detectors = {
            "repeated_phrase": self._detect_repeated_phrases,
            "numerical_trend": self._detect_numerical_trends,
            "keyword_cluster": self._detect_keyword_clusters,
            "sentiment_shift": self._detect_sentiment_shifts,
            "topic_emergence": self._detect_topic_emergence
        }
        
        logger.info("PatternRecognition initialized with %d detectors", len(self.detectors))
    
    def recognize_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main method to detect patterns in different types of data.
        
        Args:
            data: Dictionary containing various data sources to analyze
                 Expected to have keys like 'node_outputs', 'perspective_outputs', etc.
                 
        Returns:
            List of detected patterns, each as a dictionary
        """
        if not data:
            return []
        
        start_time = time.time()
        detected_patterns = []
        
        # Run each detector on the data
        for detector_name, detector_func in self.detectors.items():
            try:
                patterns = detector_func(data)
                if patterns:
                    detected_patterns.extend(patterns)
            except Exception as e:
                logger.error(f"Error in pattern detector '{detector_name}': {e}", exc_info=True)
        
        # Filter by confidence threshold
        filtered_patterns = [
            p for p in detected_patterns 
            if p.get("confidence", 0) >= self.confidence_threshold
        ]
        
        # Log pattern detection results
        duration = time.time() - start_time
        if filtered_patterns:
            logger.info(f"Detected {len(filtered_patterns)} patterns (from {len(detected_patterns)} candidates) in {duration:.3f}s")
        else:
            logger.debug(f"No patterns detected above threshold in {duration:.3f}s")
        
        # Track in history
        if filtered_patterns:
            self.pattern_history.append({
                "timestamp": time.time(),
                "pattern_count": len(filtered_patterns),
                "pattern_types": [p.get("type") for p in filtered_patterns]
            })
        
        return filtered_patterns
    
    def _detect_repeated_phrases(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detects repeated phrases in text data."""
        patterns = []
        
        # Extract text data from various sources
        text_sources = []
        
        # From node outputs
        node_outputs = data.get("node_outputs", {})
        for node_id, output in node_outputs.items():
            if isinstance(output, dict) and output.get("status") == "success":
                result = output.get("result", {})
                if isinstance(result, dict) and "summary" in result:
                    text_sources.append(result["summary"])
                elif isinstance(result, str):
                    text_sources.append(result)
        
        # From perspective outputs
        perspective_outputs = data.get("perspective_outputs", {})
        for perspective, output in perspective_outputs.items():
            if isinstance(output, str):
                text_sources.append(output)
        
        # Also check for flat_text
        if "flat_text" in data and isinstance(data["flat_text"], str):
            text_sources.append(data["flat_text"])
        
        combined_text = " ".join(text_sources)
        if not combined_text:
            return patterns
        
        # Find phrases that appear multiple times (basic implementation)
        words = combined_text.lower().split()
        if len(words) < 5:
            return patterns
        
        # Look for repeated 3-word phrases
        phrases = {}
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            phrases[phrase] = phrases.get(phrase, 0) + 1
        
        # Filter significant phrases
        for phrase, count in phrases.items():
            if count >= 2 and len(phrase) > 10:  # At least 2 occurrences and not too short
                confidence = min(1.0, count / 10)  # More occurrences = higher confidence
                patterns.append({
                    "type": "repeated_phrase",
                    "description": f"Repeated phrase: '{phrase}'",
                    "occurrences": count,
                    "confidence": confidence,
                    "timestamp": time.time()
                })
        
        return patterns
    
    def _detect_numerical_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detects trends in numerical data."""
        patterns = []
        
        # Extract numerical values from input data
        values = data.get("input_data", {}).get("values", [])
        if not values or not isinstance(values, list) or len(values) < 3:
            return patterns
        
        try:
            # Check if the values are numeric
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
            if len(numeric_values) < 3:
                return patterns
            
            # Check for increasing trend
            is_increasing = all(numeric_values[i] <= numeric_values[i+1] for i in range(len(numeric_values)-1))
            
            # Check for decreasing trend
            is_decreasing = all(numeric_values[i] >= numeric_values[i+1] for i in range(len(numeric_values)-1))
            
            if is_increasing or is_decreasing:
                # Calculate the confidence based on consistency and range
                value_range = max(numeric_values) - min(numeric_values)
                consistency = 1.0  # Perfect consistency
                confidence = min(1.0, (value_range / max(1, max(abs(v) for v in numeric_values))) * consistency)
                
                patterns.append({
                    "type": "numerical_trend",
                    "description": f"{'Increasing' if is_increasing else 'Decreasing'} numerical trend detected",
                    "values": numeric_values,
                    "direction": "increasing" if is_increasing else "decreasing",
                    "confidence": confidence,
                    "timestamp": time.time()
                })
        except (ValueError, TypeError):
            pass  # Skip if values can't be converted to float
        
        return patterns
    
    def _detect_keyword_clusters(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detects clusters of related keywords in text data."""
        # Simple implementation - could be enhanced with proper NLP techniques
        patterns = []
        text_data = ""
        
        # Extract text from data similar to _detect_repeated_phrases
        if "flat_text" in data and isinstance(data["flat_text"], str):
            text_data = data["flat_text"]
        else:
            text_sources = []
            node_outputs = data.get("node_outputs", {})
            for output in node_outputs.values():
                if isinstance(output, dict) and output.get("status") == "success":
                    result = output.get("result", {})
                    if isinstance(result, dict) and "summary" in result:
                        text_sources.append(result["summary"])
                    elif isinstance(result, str):
                        text_sources.append(result)
            text_data = " ".join(text_sources)
        
        if not text_data:
            return patterns
        
        # Define some keyword clusters (could be expanded or made dynamic)
        keyword_clusters = {
            "technology": ["ai", "computer", "algorithm", "software", "hardware", "digital", "technology", "data"],
            "nature": ["forest", "tree", "river", "mountain", "animal", "flower", "ecosystem", "natural"],
            "science": ["research", "experiment", "hypothesis", "theory", "study", "scientific", "laboratory"],
            "business": ["company", "market", "investment", "finance", "profit", "strategy", "business"]
        }
        
        # Count keywords from each cluster in the text
        text_lower = text_data.lower()
        cluster_matches = {}
        
        for cluster_name, keywords in keyword_clusters.items():
            match_count = sum(1 for keyword in keywords if keyword in text_lower)
            if match_count >= 3:  # At least 3 different keywords from the cluster
                coverage = match_count / len(keywords)
                confidence = min(1.0, coverage * 1.5)  # Higher coverage = higher confidence
                cluster_matches[cluster_name] = {
                    "matches": match_count,
                    "coverage": coverage,
                    "confidence": confidence
                }
        
        # Create patterns for matched clusters
        for cluster_name, stats in cluster_matches.items():
            if stats["confidence"] >= self.confidence_threshold:
                patterns.append({
                    "type": "keyword_cluster",
                    "description": f"Keyword cluster '{cluster_name}' detected",
                    "cluster": cluster_name,
                    "match_count": stats["matches"],
                    "confidence": stats["confidence"],
                    "timestamp": time.time()
                })
        
        return patterns
    
    def _detect_sentiment_shifts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects shifts in sentiment across multiple text sources.
        This is a simplified version - a real implementation would use more sophisticated NLP.
        """
        patterns = []
        
        # Extract sequential text chunks to analyze for sentiment shifts
        text_chunks = []
        
        # Get text chunks from node outputs in order
        node_outputs = data.get("node_outputs", {})
        for node_id, output in sorted(node_outputs.items()):
            if isinstance(output, dict) and output.get("status") == "success":
                result = output.get("result", {})
                if isinstance(result, dict) and "summary" in result:
                    text_chunks.append(result["summary"])
                elif isinstance(result, str):
                    text_chunks.append(result)
        
        if len(text_chunks) < 2:
            return patterns
        
        # Very basic sentiment analysis - using lists of positive and negative words
        positive_words = ["good", "great", "excellent", "positive", "happy", "success", "improve", "benefit"]
        negative_words = ["bad", "poor", "negative", "fail", "problem", "issue", "worsen", "damage"]
        
        # Calculate sentiment scores for each chunk
        sentiment_scores = []
        for chunk in text_chunks:
            chunk_lower = chunk.lower()
            positive_count = sum(chunk_lower.count(word) for word in positive_words)
            negative_count = sum(chunk_lower.count(word) for word in negative_words)
            
            # Calculate sentiment score: range [-1, 1]
            if positive_count == 0 and negative_count == 0:
                score = 0  # Neutral
            else:
                score = (positive_count - negative_count) / (positive_count + negative_count)
            
            sentiment_scores.append(score)
        
        # Look for significant shifts in sentiment
        for i in range(1, len(sentiment_scores)):
            shift = sentiment_scores[i] - sentiment_scores[i-1]
            if abs(shift) >= 0.5:  # Significant shift threshold
                confidence = min(1.0, abs(shift))
                direction = "positive" if shift > 0 else "negative"
                
                patterns.append({
                    "type": "sentiment_shift",
                    "description": f"Shift to {direction} sentiment detected",
                    "from_score": sentiment_scores[i-1],
                    "to_score": sentiment_scores[i],
                    "shift_magnitude": abs(shift),
                    "direction": direction,
                    "confidence": confidence,
                    "timestamp": time.time()
                })
        
        return patterns
    
    def _detect_topic_emergence(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects emergence of new topics in the conversation.
        Simplified implementation - a real system would use topic modeling.
        """
        patterns = []
        
        # Get current cycle data
        cycle = data.get("cycle", 0)
        if cycle <= 1:  # Need previous cycles for comparison
            return patterns
        
        # Simple keyword counting approach
        current_text = data.get("flat_text", "")
        if not current_text:
            return patterns
        
        # Define some topic indicators (simplified approach)
        topics = {
            "artificial_intelligence": ["ai", "artificial intelligence", "machine learning", "neural network"],
            "ethics": ["ethics", "moral", "values", "principles", "responsibility"],
            "environment": ["climate", "environment", "sustainable", "ecosystem", "green"],
            "technology": ["technology", "innovation", "digital", "software", "hardware"],
            "business": ["business", "market", "finance", "company", "investment"]
        }
        
        # Count topic keywords in the current text
        current_text_lower = current_text.lower()
        topic_counts = {}
        
        for topic, keywords in topics.items():
            count = 0
            for keyword in keywords:
                count += current_text_lower.count(keyword)
            if count > 0:
                topic_counts[topic] = count
        
        # Determine if any topics have newly emerged
        # In a real system, you'd compare with previous cycles
        # Here, just simulate with some randomness since we don't store history
        if topic_counts and random.random() < 0.3:  # 30% chance of finding a new topic
            emerging_topic = random.choice(list(topic_counts.keys()))
            confidence = min(1.0, topic_counts[emerging_topic] / 10)
            
            patterns.append({
                "type": "topic_emergence",
                "description": f"Emerging topic: '{emerging_topic}'",
                "topic": emerging_topic,
                "keyword_count": topic_counts[emerging_topic],
                "confidence": confidence,
                "timestamp": time.time()
            })
        
        return patterns
