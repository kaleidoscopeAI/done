from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict

@dataclass
class PerformanceMetric:
    """Tracks a single performance metric over time."""
    name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    weight: float = 1.0
    threshold: Optional[float] = None

class PerformanceMonitor:
    """Monitors and tracks system performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.snapshots = []
        self.alerts = []
        self.initialize_default_metrics()
        
    def initialize_default_metrics(self):
        """Initialize default performance metrics."""
        default_metrics = {
            'pattern_recognition': PerformanceMetric(
                name='pattern_recognition',
                weight=0.3,
                threshold=0.6
            ),
            'knowledge_integration': PerformanceMetric(
                name='knowledge_integration',
                weight=0.3,
                threshold=0.5
            ),
            'memory_efficiency': PerformanceMetric(
                name='memory_efficiency',
                weight=0.2,
                threshold=0.7
            ),
            'learning_speed': PerformanceMetric(
                name='learning_speed',
                weight=0.2,
                threshold=0.4
            )
        }
        
        self.metrics.update(default_metrics)
        
    def record_metrics(self, current_metrics: Dict[str, float]):
        """Record current performance metrics."""
        timestamp = datetime.now().isoformat()
        
        for metric_name, value in current_metrics.items():
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                metric.values.append(value)
                metric.timestamps.append(timestamp)
                
                # Check for threshold violations
                if metric.threshold and value < metric.threshold:
                    self._create_alert(metric_name, value, metric.threshold)
        
        # Create performance snapshot
        self._create_snapshot(current_metrics, timestamp)
        
    def get_overall_performance(self) -> float:
        """Calculate overall system performance score."""
        if not self.snapshots:
            return 0.0
            
        latest_snapshot = self.snapshots[-1]
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric in self.metrics.items():
            if metric_name in latest_snapshot['metrics']:
                total_score += latest_snapshot['metrics'][metric_name] * metric.weight
                total_weight += metric.weight
                
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def get_metric_trends(self) -> Dict[str, Dict[str, float]]:
        """Calculate trends for each metric."""
        trends = {}
        
        for metric_name, metric in self.metrics.items():
            if len(metric.values) >= 2:
                trends[metric_name] = {
                    'short_term_change': self._calculate_short_term_trend(metric.values),
                    'long_term_trend': self._calculate_long_term_trend(metric.values),
                    'volatility': self._calculate_volatility(metric.values)
                }
                
        return trends
        
    def _calculate_short_term_trend(self, values: List[float], window: int = 5) -> float:
        """Calculate short-term trend using recent values."""
        if len(values) < 2:
            return 0.0
            
        recent_values = values[-min(window, len(values)):]
        if len(recent_values) < 2:
            return 0.0
            
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
        
    def _calculate_long_term_trend(self, values: List[float]) -> float:
        """Calculate long-term trend using all values."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
        
    def _calculate_volatility(self, values: List[float], window: int = 10) -> float:
        """Calculate metric volatility."""
        if len(values) < 2:
            return 0.0
            
        recent_values = values[-min(window, len(values)):]
        return np.std(recent_values)
        
    def _create_snapshot(self, metrics: Dict[str, float], timestamp: str):
        """Create a performance snapshot."""
        snapshot = {
            'timestamp': timestamp,
            'metrics': metrics.copy(),
            'overall_score': self.get_overall_performance(),
            'trends': self.get_metric_trends()
        }
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots (last 100)
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
            
    def _create_alert(self, metric_name: str, value: float, threshold: float):
        """Create performance alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric_name,
            'value': value,
            'threshold': threshold,
            'status': 'active'
        }
        
        self.alerts.append(alert)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.snapshots:
            return {}
            
        latest_snapshot = self.snapshots[-1]
        trends = self.get_metric_trends()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_performance': self.get_overall_performance(),
            'metrics': {
                name: {
                    'current_value': latest_snapshot['metrics'].get(name, 0.0),
                    'trend': trends.get(name, {}),
                    'threshold': metric.threshold,
                    'weight': metric.weight
                }
                for name, metric in self.metrics.items()
            },
            'alerts': [
                alert for alert in self.alerts 
                if alert['status'] == 'active'
            ],
            'recommendations': self._generate_recommendations(trends)
        }
        
        return report
        
    def _generate_recommendations(self, trends: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        for metric_name, trend_data in trends.items():
            if trend_data['short_term_change'] < 0:
                recommendations.append({
                    'metric': metric_name,
                    'severity': 'high' if self.metrics[metric_name].threshold else 'medium',
                    'trend': 'declining',
                    'suggestion': f"Investigate declining {metric_name} performance"
                })
            elif trend_data['volatility'] > 0.2:
                recommendations.append({
                    'metric': metric_name,
                    'severity': 'medium',
                    'trend': 'unstable',
                    'suggestion': f"Stabilize {metric_name} performance"
                })
                
        return recommendations
        
    def get_metric_distribution(self, metric_name: str) -> Dict[str, float]:
        """Get statistical distribution of a metric's values."""
        if metric_name not in self.metrics:
            return {}
            
        values = self.metrics[metric_name].values
        if not values:
            return {}
            
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'quartiles': [
                float(np.percentile(values, 25)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 75))
            ]
        }
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active performance alerts."""
        return [
            alert for alert in self.alerts 
            if alert['status'] == 'active'
        ]
        
    def acknowledge_alert(self, alert_id: int):
        """Acknowledge and resolve a performance alert."""
        if 0 <= alert_id < len(self.alerts):
            self.alerts[alert_id]['status'] = 'resolved'
