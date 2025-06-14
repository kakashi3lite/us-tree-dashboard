from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary
from src.monitoring.logger import logger

# Request metrics
HTTP_REQUEST_TOTAL = Counter(
    'http_request_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

# Database metrics
DB_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

ACTIVE_DB_CONNECTIONS = Gauge(
    'active_db_connections',
    'Number of active database connections'
)

# Geospatial metrics
GEOSPATIAL_QUERY_DURATION = Histogram(
    'geospatial_query_duration_seconds',
    'Geospatial query duration in seconds',
    ['operation_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

GEOSPATIAL_RESULT_COUNT = Summary(
    'geospatial_result_count',
    'Number of results returned by geospatial queries',
    ['operation_type']
)

# ML metrics
ML_PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'Machine learning prediction duration in seconds',
    ['model_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ML_PREDICTION_ERROR = Counter(
    'ml_prediction_error_total',
    'Total number of ML prediction errors',
    ['model_type', 'error_type']
)

@dataclass
class RequestMetrics:
    start_time: datetime
    method: str
    endpoint: str
    status_code: Optional[int] = None
    error: Optional[str] = None

    def record(self):
        duration = (datetime.now() - self.start_time).total_seconds()

        # Record request count
        HTTP_REQUEST_TOTAL.labels(
            method=self.method,
            endpoint=self.endpoint,
            status=self.status_code or 500
        ).inc()

        # Record request duration
        HTTP_REQUEST_DURATION.labels(
            method=self.method,
            endpoint=self.endpoint
        ).observe(duration)

        # Log metrics
        logger.info(
            'Request Metrics',
            extra={
                'metric_type': 'request',
                'method': self.method,
                'endpoint': self.endpoint,
                'status_code': self.status_code,
                'duration': duration,
                'error': self.error
            }
        )

class DatabaseMetrics:
    @staticmethod
    def record_query(query_type: str, duration: float):
        DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)

    @staticmethod
    def update_connections(count: int):
        ACTIVE_DB_CONNECTIONS.set(count)

    @staticmethod
    def log_slow_query(query_type: str, duration: float, query: str):
        if duration > 1.0:  # Log queries taking more than 1 second
            logger.warning(
                'Slow Query Detected',
                extra={
                    'metric_type': 'database',
                    'query_type': query_type,
                    'duration': duration,
                    'query': query
                }
            )

class GeospatialMetrics:
    @staticmethod
    def record_query(operation_type: str, duration: float, result_count: int):
        GEOSPATIAL_QUERY_DURATION.labels(
            operation_type=operation_type
        ).observe(duration)

        GEOSPATIAL_RESULT_COUNT.labels(
            operation_type=operation_type
        ).observe(result_count)

        logger.info(
            'Geospatial Query Metrics',
            extra={
                'metric_type': 'geospatial',
                'operation_type': operation_type,
                'duration': duration,
                'result_count': result_count
            }
        )

class MLMetrics:
    @staticmethod
    def record_prediction(model_type: str, duration: float, error: Optional[str] = None):
        ML_PREDICTION_DURATION.labels(model_type=model_type).observe(duration)

        if error:
            ML_PREDICTION_ERROR.labels(
                model_type=model_type,
                error_type=type(error).__name__
            ).inc()

            logger.error(
                'ML Prediction Error',
                extra={
                    'metric_type': 'ml',
                    'model_type': model_type,
                    'error': str(error),
                    'duration': duration
                }
            )
        else:
            logger.info(
                'ML Prediction Success',
                extra={
                    'metric_type': 'ml',
                    'model_type': model_type,
                    'duration': duration
                }
            )

class CacheMetrics:
    @staticmethod
    def record_cache_hit(cache_type: str):
        CACHE_HITS.labels(cache_type=cache_type).inc()

    @staticmethod
    def record_cache_miss(cache_type: str):
        CACHE_MISSES.labels(cache_type=cache_type).inc()

    @staticmethod
    def log_cache_stats(cache_type: str, hit_rate: float):
        logger.info(
            'Cache Statistics',
            extra={
                'metric_type': 'cache',
                'cache_type': cache_type,
                'hit_rate': hit_rate
            }
        )

def get_metrics_summary() -> Dict[str, List[Dict]]:
    """
    Generate a summary of all metrics for monitoring dashboards
    """
    return {
        'request_metrics': [
            {
                'name': 'http_request_total',
                'type': 'counter',
                'description': 'Total number of HTTP requests',
                'labels': ['method', 'endpoint', 'status']
            },
            {
                'name': 'http_request_duration_seconds',
                'type': 'histogram',
                'description': 'HTTP request duration in seconds',
                'labels': ['method', 'endpoint']
            }
        ],
        'cache_metrics': [
            {
                'name': 'cache_hits_total',
                'type': 'counter',
                'description': 'Total number of cache hits',
                'labels': ['cache_type']
            },
            {
                'name': 'cache_misses_total',
                'type': 'counter',
                'description': 'Total number of cache misses',
                'labels': ['cache_type']
            }
        ],
        'database_metrics': [
            {
                'name': 'db_query_duration_seconds',
                'type': 'histogram',
                'description': 'Database query duration in seconds',
                'labels': ['query_type']
            },
            {
                'name': 'active_db_connections',
                'type': 'gauge',
                'description': 'Number of active database connections',
                'labels': []
            }
        ],
        'geospatial_metrics': [
            {
                'name': 'geospatial_query_duration_seconds',
                'type': 'histogram',
                'description': 'Geospatial query duration in seconds',
                'labels': ['operation_type']
            },
            {
                'name': 'geospatial_result_count',
                'type': 'summary',
                'description': 'Number of results returned by geospatial queries',
                'labels': ['operation_type']
            }
        ],
        'ml_metrics': [
            {
                'name': 'ml_prediction_duration_seconds',
                'type': 'histogram',
                'description': 'Machine learning prediction duration in seconds',
                'labels': ['model_type']
            },
            {
                'name': 'ml_prediction_error_total',
                'type': 'counter',
                'description': 'Total number of ML prediction errors',
                'labels': ['model_type', 'error_type']
            }
        ]
    }
