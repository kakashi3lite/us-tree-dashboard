"""
Performance monitoring metrics.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
from dataclasses import dataclass, field
import psutil
import time
import numpy as np
from statistics import mean, median, stdev
from .base import BaseMetric, MetricSeries

@dataclass
class PerformanceMetric(BaseMetric):
    """System performance metric.

    Defaults ensure compatibility with BaseMetric field ordering.
    """
    component: str = ""
    metric_type: str = ""
    percentiles: Optional[Dict[str, float]] = None

    def validate(self) -> bool:
        """Validate performance metric value."""
        if self.metric_type in ['latency', 'memory', 'cpu']:
            return isinstance(self.value, (int, float)) and self.value >= 0
        return super().validate()

@dataclass
class SystemPerformanceMetrics:
    """Collection of system performance metrics."""
    timestamp: datetime.datetime
    cpu_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]
    disk_metrics: Dict[str, float]
    latency_metrics: Optional[Dict[str, Dict[str, float]]] = None
    error_rates: Optional[Dict[str, float]] = None

    def to_metric_list(self) -> List[PerformanceMetric]:
        """Convert system metrics to list of individual metrics."""
        metrics = []
        timestamp = self.timestamp

        # CPU metrics
        for name, value in self.cpu_metrics.items():
            metrics.append(PerformanceMetric(
                name=f"cpu_{name}",
                value=value,
                timestamp=timestamp,
                tags={'subsystem': 'cpu'},
                component='system',
                metric_type='cpu'
            ))

        # Memory metrics
        for name, value in self.memory_metrics.items():
            metrics.append(PerformanceMetric(
                name=f"memory_{name}",
                value=value,
                timestamp=timestamp,
                tags={'subsystem': 'memory'},
                component='system',
                metric_type='memory'
            ))

        # Disk metrics
        for name, value in self.disk_metrics.items():
            metrics.append(PerformanceMetric(
                name=f"disk_{name}",
                value=value,
                timestamp=timestamp,
                tags={'subsystem': 'disk'},
                component='system',
                metric_type='disk'
            ))

        # Latency metrics
        if self.latency_metrics:
            for component, latencies in self.latency_metrics.items():
                for metric_name, value in latencies.items():
                    metrics.append(PerformanceMetric(
                        name=f"latency_{component}_{metric_name}",
                        value=value,
                        timestamp=timestamp,
                        tags={'subsystem': 'latency', 'component': component},
                        component=component,
                        metric_type='latency'
                    ))

        # Error rates
        if self.error_rates:
            for component, rate in self.error_rates.items():
                metrics.append(PerformanceMetric(
                    name=f"error_rate_{component}",
                    value=rate,
                    timestamp=timestamp,
                    tags={'subsystem': 'errors'},
                    component=component,
                    metric_type='error'
                ))

        return metrics

def collect_system_metrics() -> SystemPerformanceMetrics:
    """Collect current system performance metrics."""
    timestamp = datetime.datetime.now()
    
    # CPU metrics
    cpu_metrics = {
        'percent': psutil.cpu_percent(interval=1),
        'count': psutil.cpu_count(),
        'load_1min': psutil.getloadavg()[0]
    }
    
    # Memory metrics
    memory = psutil.virtual_memory()
    memory_metrics = {
        'total_gb': memory.total / (1024 ** 3),
        'available_gb': memory.available / (1024 ** 3),
        'percent_used': memory.percent
    }
    
    # Disk metrics
    disk = psutil.disk_usage('/')
    disk_metrics = {
        'total_gb': disk.total / (1024 ** 3),
        'free_gb': disk.free / (1024 ** 3),
        'percent_used': disk.percent
    }
    
    return SystemPerformanceMetrics(
        timestamp=timestamp,
        cpu_metrics=cpu_metrics,
        memory_metrics=memory_metrics,
        disk_metrics=disk_metrics
    )

def collect_latency_metrics(latencies: List[float],
                          component: str,
                          include_percentiles: bool = True) -> Dict[str, float]:
    """Calculate latency metrics from a list of measurements."""
    if not latencies:
        return {}

    metrics = {
        'mean': mean(latencies),
        'median': median(latencies),
        'min': min(latencies),
        'max': max(latencies)
    }

    if len(latencies) > 1:
        metrics['std'] = stdev(latencies)

    if include_percentiles:
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'p{p}'] = np.percentile(latencies, p)

    return metrics

class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self, metrics_store):
        """Initialize performance monitor."""
        self.metrics_store = metrics_store
        self.start_time = time.time()
        self.latencies = {}
        self.error_counts = {}
        
    def record_latency(self, component: str, latency: float):
        """Record a latency measurement."""
        if component not in self.latencies:
            self.latencies[component] = []
        self.latencies[component].append(latency)
        
    def record_error(self, component: str):
        """Record an error occurrence."""
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        
    def get_error_rates(self) -> Dict[str, float]:
        """Calculate error rates per component."""
        elapsed_time = time.time() - self.start_time
        hours = max(elapsed_time / 3600, 1)  # At least 1 hour to avoid division by zero
        
        return {
            component: count / hours
            for component, count in self.error_counts.items()
        }
        
    def collect_metrics(self) -> SystemPerformanceMetrics:
        """Collect all performance metrics."""
        system_metrics = collect_system_metrics()
        
        # Add latency metrics
        latency_metrics = {}
        for component, measurements in self.latencies.items():
            latency_metrics[component] = collect_latency_metrics(measurements, component)
        
        # Add error rates
        error_rates = self.get_error_rates()
        
        # Update the system metrics with latency and error data
        system_metrics.latency_metrics = latency_metrics
        system_metrics.error_rates = error_rates
        
        return system_metrics
