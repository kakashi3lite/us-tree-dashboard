"""
Metrics database storage and retrieval.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import datetime
import numpy as np
from .base import BaseMetric, MetricSeries, MetricConfig
from .model_metrics import ModelMetric, ModelTrainingMetrics, ModelPredictionMetrics
from .environmental_metrics import EnvironmentalMetric, EnvironmentalImpactMetrics
from .performance_metrics import PerformanceMetric, SystemPerformanceMetrics

class MetricsStore:
    """Persistent storage for metrics with JSON serialization."""
    
    def __init__(self, base_path: Optional[Path] = None, config: Optional[MetricConfig] = None):
        """Initialize metrics store."""
        self.base_path = base_path or Path(__file__).parent.parent.parent / "metrics"
        self.base_path.mkdir(exist_ok=True)
        
        # Create metric type subdirectories
        self.model_metrics_path = self.base_path / "model_metrics"
        self.environmental_metrics_path = self.base_path / "environmental_metrics"
        self.performance_metrics_path = self.base_path / "performance_metrics"
        
        for path in [self.model_metrics_path, self.environmental_metrics_path,
                    self.performance_metrics_path]:
            path.mkdir(exist_ok=True)

        # Store configuration
        self.config = config or MetricConfig()

    def _save_metrics(self, metrics: List[BaseMetric], directory: Path):
        """Save metrics to JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = directory / f"metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)

    def save_model_metrics(self, metrics: Union[ModelTrainingMetrics, ModelPredictionMetrics]):
        """Save ML model metrics."""
        self._save_metrics(metrics.to_metric_list(), self.model_metrics_path)

    def save_environmental_metrics(self, metrics: EnvironmentalImpactMetrics):
        """Save environmental impact metrics."""
        self._save_metrics(metrics.to_metric_list(), self.environmental_metrics_path)

    def save_performance_metrics(self, metrics: SystemPerformanceMetrics):
        """Save system performance metrics."""
        self._save_metrics(metrics.to_metric_list(), self.performance_metrics_path)

    def get_metrics(self, 
                   directory: Path,
                   start_time: Optional[datetime.datetime] = None,
                   end_time: Optional[datetime.datetime] = None,
                   metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve metrics from storage with optional filtering."""
        metrics = []
        
        for file in sorted(directory.glob("metrics_*.json")):
            with open(file) as f:
                file_metrics = json.load(f)
                
            for metric in file_metrics:
                metric_time = datetime.datetime.fromisoformat(metric['timestamp'])
                
                if start_time and metric_time < start_time:
                    continue
                if end_time and metric_time > end_time:
                    continue
                if metric_name and metric['name'] != metric_name:
                    continue
                
                metrics.append(metric)
        
        return metrics

    def get_model_metrics(self, 
                         model_name: Optional[str] = None,
                         metric_type: Optional[str] = None,
                         **kwargs) -> List[Dict[str, Any]]:
        """Get ML model metrics with optional filtering."""
        metrics = self.get_metrics(self.model_metrics_path, **kwargs)
        
        if model_name:
            metrics = [m for m in metrics if m.get('model_name') == model_name]
        if metric_type:
            metrics = [m for m in metrics if m.get('metric_type') == metric_type]
            
        return metrics

    def get_environmental_metrics(self, 
                                impact_type: Optional[str] = None,
                                **kwargs) -> List[Dict[str, Any]]:
        """Get environmental metrics with optional filtering."""
        metrics = self.get_metrics(self.environmental_metrics_path, **kwargs)
        
        if impact_type:
            metrics = [m for m in metrics if m.get('impact_type') == impact_type]
            
        return metrics

    def get_performance_metrics(self,
                              component: Optional[str] = None,
                              metric_type: Optional[str] = None,
                              **kwargs) -> List[Dict[str, Any]]:
        """Get performance metrics with optional filtering."""
        metrics = self.get_metrics(self.performance_metrics_path, **kwargs)
        
        if component:
            metrics = [m for m in metrics if m.get('component') == component]
        if metric_type:
            metrics = [m for m in metrics if m.get('metric_type') == metric_type]
            
        return metrics

    def get_metric_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate trend analysis for a set of metrics."""
        import numpy as np
        
        if not metrics:
            return {}
        
        # Group metrics by name
        grouped_metrics = {}
        for metric in metrics:
            name = metric['name']
            if name not in grouped_metrics:
                grouped_metrics[name] = []
            grouped_metrics[name].append(metric)
        
        trends = {}
        for name, metric_group in grouped_metrics.items():
            values = [m['value'] for m in metric_group]
            if not isinstance(values[0], (int, float)):
                continue
                
            values_arr = np.array(values)
            
            trends[name] = {
                'mean': float(np.mean(values_arr)),
                'std': float(np.std(values_arr)),
                'min': float(np.min(values_arr)),
                'max': float(np.max(values_arr)),
                'trend_direction': 'increasing' if len(values) > 1 and values[-1] > values[0]
                                else 'decreasing' if len(values) > 1 and values[-1] < values[0]
                                else 'stable'
            }
            
        return trends

    def export_metrics_report(self, output_file: Path):
        """Generate comprehensive metrics report."""
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'models': {},
            'environmental_impact': {},
            'performance': {}
        }
        
        # Get model metrics trends
        model_metrics = self.get_model_metrics()
        if model_metrics:
            by_model = {}
            for metric in model_metrics:
                model_name = metric.get('model_name', 'unknown')
                if model_name not in by_model:
                    by_model[model_name] = []
                by_model[model_name].append(metric)
            
            for model_name, metrics in by_model.items():
                report['models'][model_name] = {
                    'latest_metrics': sorted(metrics, key=lambda x: x['timestamp'])[-1],
                    'historical_trends': self.get_metric_trends(metrics),
                    'prediction_latencies': self.get_metric_trends(
                        [m for m in metrics if m.get('metric_type') == 'latency']
                    )
                }
        
        # Get environmental metrics trends
        env_metrics = self.get_environmental_metrics()
        if env_metrics:
            env_impact = sorted(env_metrics, key=lambda x: x['timestamp'])[-1]
            report['environmental_impact'] = {
                'latest_metrics': env_impact,
                'historical_trends': self.get_metric_trends(env_metrics),
                'confidence_intervals': {
                    m['name']: m.get('confidence_intervals', {})
                    for m in env_metrics
                    if m.get('confidence_intervals')
                }
            }
        
        # Get performance metrics trends
        perf_metrics = self.get_performance_metrics()
        if perf_metrics:
            report['performance'] = {
                'system_metrics': self.get_metric_trends(
                    [m for m in perf_metrics if m.get('component') == 'system']
                ),
                'latency_trends': self.get_metric_trends(
                    [m for m in perf_metrics if m.get('metric_type') == 'latency']
                ),
                'error_rates': self.get_metric_trends(
                    [m for m in perf_metrics if m.get('metric_type') == 'error']
                )
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

# Global metrics store instance
metrics_store = MetricsStore()
