from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from prometheus_client import Counter, Histogram, Gauge
import json
import os

@dataclass
class MetricsConfig:
    app_name: str
    environment: str
    enable_prometheus: bool = True

class MetricsCollector:
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'logs',
            'metrics.json'
        )

        # Prometheus metrics
        if self.config.enable_prometheus:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status']
        )

        self.request_duration = Histogram(
            'request_duration_seconds',
            'Request duration in seconds',
            ['endpoint']
        )

        self.active_users = Gauge(
            'active_users',
            'Number of active users'
        )

        self.query_duration = Histogram(
            'query_duration_seconds',
            'Database query duration in seconds',
            ['query_type']
        )

        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )

        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )

    def record_api_request(self, endpoint: str, method: str, status: str, duration: float) -> None:
        """Record API request metrics."""
        if self.config.enable_prometheus:
            self.api_requests.labels(endpoint=endpoint, method=method, status=status).inc()
            self.request_duration.labels(endpoint=endpoint).observe(duration)

        self._save_metric({
            'type': 'api_request',
            'endpoint': endpoint,
            'method': method,
            'status': status,
            'duration': duration,
            'timestamp': datetime.utcnow().isoformat()
        })

    def record_query_performance(self, query_type: str, duration: float) -> None:
        """Record database query performance metrics."""
        if self.config.enable_prometheus:
            self.query_duration.labels(query_type=query_type).observe(duration)

        self._save_metric({
            'type': 'query_performance',
            'query_type': query_type,
            'duration': duration,
            'timestamp': datetime.utcnow().isoformat()
        })

    def record_cache_operation(self, cache_type: str, hit: bool) -> None:
        """Record cache operation metrics."""
        if self.config.enable_prometheus:
            if hit:
                self.cache_hits.labels(cache_type=cache_type).inc()
            else:
                self.cache_misses.labels(cache_type=cache_type).inc()

        self._save_metric({
            'type': 'cache_operation',
            'cache_type': cache_type,
            'hit': hit,
            'timestamp': datetime.utcnow().isoformat()
        })

    def update_active_users(self, count: int) -> None:
        """Update active users count."""
        if self.config.enable_prometheus:
            self.active_users.set(count)

        self._save_metric({
            'type': 'active_users',
            'count': count,
            'timestamp': datetime.utcnow().isoformat()
        })

    def _save_metric(self, metric: Dict[str, Any]) -> None:
        """Save metric to JSON file."""
        try:
            metrics = self._load_metrics()
            metrics.append(metric)
            
            # Keep only last 1000 metrics
            if len(metrics) > 1000:
                metrics = metrics[-1000:]

            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f)
        except Exception as e:
            print(f"Error saving metric: {e}")

    def _load_metrics(self) -> List[Dict[str, Any]]:
        """Load metrics from JSON file."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
        return []

    def get_metrics_summary(self, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        metrics = self._load_metrics()
        
        if metric_type:
            metrics = [m for m in metrics if m['type'] == metric_type]

        return {
            'total_records': len(metrics),
            'latest_timestamp': metrics[-1]['timestamp'] if metrics else None,
            'metrics_by_type': self._group_metrics_by_type(metrics)
        }

    def _group_metrics_by_type(self, metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group metrics by type and count occurrences."""
        type_counts = {}
        for metric in metrics:
            metric_type = metric['type']
            type_counts[metric_type] = type_counts.get(metric_type, 0) + 1
        return type_counts