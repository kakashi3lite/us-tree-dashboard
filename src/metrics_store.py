"""
Metrics storage and tracking system for ML models and performance monitoring.

This module provides centralized metrics storage and tracking for:
1. ML model performance metrics
2. Model training history
3. Environmental impact calculations
4. System performance metrics
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class ModelMetrics:
    """Stores metrics for ML model performance."""
    model_name: str
    version: str
    timestamp: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    dataset_size: int
    training_time: float

@dataclass
class EnvironmentalMetrics:
    """Stores environmental impact metrics."""
    timestamp: str
    total_trees: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, tuple]
    species_baselines: Dict[str, Dict[str, float]]

class MetricsStore:
    """Central storage for all metrics."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize metrics store."""
        self.base_path = base_path or Path(__file__).parent.parent / "metrics"
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories for different metric types
        self.model_metrics_path = self.base_path / "model_metrics"
        self.environmental_metrics_path = self.base_path / "environmental_metrics"
        self.performance_metrics_path = self.base_path / "performance_metrics"
        
        for path in [self.model_metrics_path, self.environmental_metrics_path, 
                    self.performance_metrics_path]:
            path.mkdir(exist_ok=True)

    def save_model_metrics(self, metrics: ModelMetrics):
        """Save ML model metrics."""
        metrics_file = self.model_metrics_path / f"{metrics.model_name}_{metrics.version}.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    def get_model_metrics_history(self, model_name: str) -> List[ModelMetrics]:
        """Get historical metrics for a model."""
        metrics_list = []
        for file in self.model_metrics_path.glob(f"{model_name}_*.json"):
            with open(file) as f:
                data = json.load(f)
                metrics_list.append(ModelMetrics(**data))
        return sorted(metrics_list, key=lambda x: x.timestamp)

    def save_environmental_metrics(self, metrics: EnvironmentalMetrics):
        """Save environmental impact metrics."""
        date_str = metrics.timestamp.split('T')[0]  # Get date part only
        metrics_file = self.environmental_metrics_path / f"env_metrics_{date_str}.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    def get_environmental_metrics_history(self, 
                                       start_date: Optional[str] = None, 
                                       end_date: Optional[str] = None) -> List[EnvironmentalMetrics]:
        """Get historical environmental metrics."""
        metrics_list = []
        for file in self.environmental_metrics_path.glob("env_metrics_*.json"):
            with open(file) as f:
                data = json.load(f)
                metrics = EnvironmentalMetrics(**data)
                
                # Filter by date range if specified
                if start_date and metrics.timestamp < start_date:
                    continue
                if end_date and metrics.timestamp > end_date:
                    continue
                    
                metrics_list.append(metrics)
        return sorted(metrics_list, key=lambda x: x.timestamp)

    def calculate_metrics_trend(self, metrics_list: List[Union[ModelMetrics, EnvironmentalMetrics]]) -> Dict[str, dict]:
        """Calculate trend analysis for metrics."""
        if not metrics_list:
            return {}
        
        # For model metrics, analyze performance trends
        if isinstance(metrics_list[0], ModelMetrics):
            timestamps = [m.timestamp for m in metrics_list]
            metrics_values = {k: [m.metrics[k] for m in metrics_list] 
                            for k in metrics_list[0].metrics.keys()}
        # For environmental metrics, analyze impact trends
        else:
            timestamps = [m.timestamp for m in metrics_list]
            metrics_values = {k: [m.metrics[k] for m in metrics_list] 
                            for k in metrics_list[0].metrics.keys()}

        trends = {}
        for metric_name, values in metrics_values.items():
            values_arr = np.array(values)
            trends[metric_name] = {
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
            'generated_at': datetime.now().isoformat(),
            'models': {},
            'environmental_impact': {}
        }
        
        # Gather all model metrics
        model_names = set()
        for file in self.model_metrics_path.glob("*.json"):
            model_name = file.stem.split('_')[0]
            model_names.add(model_name)
            
        for model_name in model_names:
            history = self.get_model_metrics_history(model_name)
            if history:
                report['models'][model_name] = {
                    'latest_metrics': asdict(history[-1]),
                    'historical_trends': self.calculate_metrics_trend(history)
                }
        
        # Gather environmental metrics
        env_history = self.get_environmental_metrics_history()
        if env_history:
            report['environmental_impact'] = {
                'latest_metrics': asdict(env_history[-1]),
                'historical_trends': self.calculate_metrics_trend(env_history)
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

# Global metrics store instance
metrics_store = MetricsStore()
