"""Feature flag enabled metrics store with stub fallback.

This module provides metrics storage functionality with feature flags
to avoid import failures while maintaining coverage parsing capability.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

# Feature flag for heavy dependencies
ENABLE_METRICS_FEATURES = os.getenv('METRICS_FULL_FEATURES', 'false').lower() == 'true'

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if ENABLE_METRICS_FEATURES:
    try:
        import pandas as pd
        import numpy as np
        METRICS_DEPENDENCIES_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Metrics features disabled due to missing dependencies: {e}")
        METRICS_DEPENDENCIES_AVAILABLE = False
else:
    METRICS_DEPENDENCIES_AVAILABLE = False


@dataclass
class MetricConfig:
    """Configuration for metrics storage."""
    retention_days: int = 30
    max_metrics_per_file: int = 1000
    auto_cleanup: bool = True


@dataclass 
class BaseMetric:
    """Base metric structure."""
    name: str
    value: Union[float, int, str]
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags or {}
        }


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
        if not metrics:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"
        filepath = directory / filename
        
        # Convert metrics to dictionaries
        metrics_data = [metric.to_dict() for metric in metrics]
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"Saved {len(metrics)} metrics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def save_model_metrics(self, metrics):
        """Save ML model metrics."""
        if isinstance(metrics, dict):
            # Convert dict to BaseMetric for consistency
            metric_obj = BaseMetric(
                name=metrics.get('name', 'model_metric'),
                value=metrics.get('value', 0),
                timestamp=datetime.now(),
                tags=metrics.get('tags', {})
            )
            metrics = [metric_obj]
        elif not isinstance(metrics, list):
            metrics = [metrics]
            
        self._save_metrics(metrics, self.model_metrics_path)
    
    def save_environmental_metrics(self, metrics):
        """Save environmental impact metrics."""
        if isinstance(metrics, dict):
            metric_obj = BaseMetric(
                name=metrics.get('name', 'environmental_metric'),
                value=metrics.get('value', 0),
                timestamp=datetime.now(),
                tags=metrics.get('tags', {})
            )
            metrics = [metric_obj]
        elif not isinstance(metrics, list):
            metrics = [metrics]
            
        self._save_metrics(metrics, self.environmental_metrics_path)
    
    def save_performance_metrics(self, metrics):
        """Save system performance metrics."""
        if isinstance(metrics, dict):
            metric_obj = BaseMetric(
                name=metrics.get('name', 'performance_metric'),
                value=metrics.get('value', 0),
                timestamp=datetime.now(),
                tags=metrics.get('tags', {})
            )
            metrics = [metric_obj]
        elif not isinstance(metrics, list):
            metrics = [metrics]
            
        self._save_metrics(metrics, self.performance_metrics_path)
    
    def get_metrics(self, 
                   directory: Path,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve metrics from storage with optional filtering."""
        metrics = []
        
        if not directory.exists():
            return metrics
        
        for file in sorted(directory.glob("metrics_*.json")):
            try:
                with open(file) as f:
                    file_metrics = json.load(f)
                    
                for metric in file_metrics:
                    try:
                        metric_time = datetime.fromisoformat(metric['timestamp'])
                    except (ValueError, KeyError):
                        continue
                    
                    if start_time and metric_time < start_time:
                        continue
                    if end_time and metric_time > end_time:
                        continue
                    if metric_name and metric.get('name') != metric_name:
                        continue
                    
                    metrics.append(metric)
            except Exception as e:
                logger.warning(f"Failed to read metrics file {file}: {e}")
        
        return metrics
    
    def get_model_metrics(self, 
                         model_name: Optional[str] = None,
                         metric_type: Optional[str] = None,
                         **kwargs) -> List[Dict[str, Any]]:
        """Get ML model metrics with optional filtering."""
        metrics = self.get_metrics(self.model_metrics_path, **kwargs)
        
        if model_name:
            metrics = [m for m in metrics if m.get('tags', {}).get('model_name') == model_name]
        if metric_type:
            metrics = [m for m in metrics if m.get('tags', {}).get('metric_type') == metric_type]
            
        return metrics

    def get_environmental_metrics(self, 
                                impact_type: Optional[str] = None,
                                **kwargs) -> List[Dict[str, Any]]:
        """Get environmental metrics with optional filtering."""
        metrics = self.get_metrics(self.environmental_metrics_path, **kwargs)
        
        if impact_type:
            metrics = [m for m in metrics if m.get('tags', {}).get('impact_type') == impact_type]
            
        return metrics
    
    def get_performance_metrics(self,
                              component: Optional[str] = None,
                              metric_type: Optional[str] = None,
                              **kwargs) -> List[Dict[str, Any]]:
        """Get performance metrics with optional filtering."""
        metrics = self.get_metrics(self.performance_metrics_path, **kwargs)
        
        if component:
            metrics = [m for m in metrics if m.get('tags', {}).get('component') == component]
        if metric_type:
            metrics = [m for m in metrics if m.get('tags', {}).get('metric_type') == metric_type]
            
        return metrics
    
    def get_metric_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate trend analysis for a set of metrics."""
        if not metrics:
            return {}
        
        trends = {}
        
        # Group metrics by name
        metric_groups = {}
        for metric in metrics:
            name = metric.get('name', 'unknown')
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric)
        
        # Calculate trends for each metric group
        for name, group in metric_groups.items():
            if len(group) < 2:
                trends[name] = {"trend": "insufficient_data", "change": 0.0}
                continue
                
            # Sort by timestamp
            sorted_group = sorted(group, key=lambda x: x.get('timestamp', ''))
            
            if not sorted_group:
                continue
            
            try:
                first_value = float(sorted_group[0].get('value', 0))
                last_value = float(sorted_group[-1].get('value', 0))
                
                if first_value == 0:
                    change_percent = 0.0
                else:
                    change_percent = ((last_value - first_value) / first_value) * 100
                
                if change_percent > 5:
                    trend = "increasing"
                elif change_percent < -5:
                    trend = "decreasing"
                else:
                    trend = "stable"
                    
                trends[name] = {
                    "trend": trend,
                    "change": change_percent,
                    "first_value": first_value,
                    "last_value": last_value,
                    "data_points": len(group)
                }
            except (ValueError, TypeError):
                trends[name] = {"trend": "invalid_data", "change": 0.0}
        
        return trends
    
    def export_metrics_report(self, output_file: Path):
        """Generate comprehensive metrics report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "model_metrics": [],
            "environmental_metrics": [],
            "performance_metrics": []
        }
        
        # Collect all metrics
        model_metrics = self.get_model_metrics()
        env_metrics = self.get_environmental_metrics()
        perf_metrics = self.get_performance_metrics()
        
        report["model_metrics"] = model_metrics
        report["environmental_metrics"] = env_metrics
        report["performance_metrics"] = perf_metrics
        
        # Calculate summary
        report["summary"] = {
            "total_model_metrics": len(model_metrics),
            "total_environmental_metrics": len(env_metrics),
            "total_performance_metrics": len(perf_metrics),
            "total_metrics": len(model_metrics) + len(env_metrics) + len(perf_metrics)
        }
        
        # Add trends
        all_metrics = model_metrics + env_metrics + perf_metrics
        trends = self.get_metric_trends(all_metrics)
        report["trends"] = trends
        
        # Save report
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Exported metrics report to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
    
    def cleanup_old_metrics(self, days: Optional[int] = None):
        """Clean up metrics older than specified days."""
        cleanup_days = days or self.config.retention_days
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)
        
        for directory in [self.model_metrics_path, self.environmental_metrics_path, 
                         self.performance_metrics_path]:
            if not directory.exists():
                continue
                
            for file in directory.glob("metrics_*.json"):
                try:
                    # Parse date from filename
                    filename = file.stem
                    date_part = filename.replace("metrics_", "")
                    file_date = datetime.strptime(date_part, "%Y%m%d_%H%M%S")
                    
                    if file_date < cutoff_date:
                        file.unlink()
                        logger.info(f"Cleaned up old metrics file: {file}")
                except Exception as e:
                    logger.warning(f"Could not cleanup file {file}: {e}")


# Example usage
def main():
    """Main function for testing."""
    logger.info(f"Metrics Store - Dependencies Available: {METRICS_DEPENDENCIES_AVAILABLE}")
    
    # Create metrics store
    store = MetricsStore()
    
    # Test saving metrics
    test_metric = BaseMetric(
        name="test_accuracy",
        value=0.85,
        timestamp=datetime.now(),
        tags={"model_name": "tree_classifier", "metric_type": "accuracy"}
    )
    
    store.save_model_metrics([test_metric])
    
    # Test retrieving metrics
    metrics = store.get_model_metrics()
    logger.info(f"Retrieved {len(metrics)} model metrics")
    
    # Test trends
    trends = store.get_metric_trends(metrics)
    logger.info(f"Calculated trends: {trends}")


if __name__ == "__main__":
    main()