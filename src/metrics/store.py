"""Metrics database storage and retrieval with Redis caching and PostGIS integration."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import datetime
import numpy as np
import redis
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry
from .base import BaseMetric, MetricSeries, MetricConfig
from .model_metrics import ModelMetric, ModelTrainingMetrics, ModelPredictionMetrics
from .environmental_metrics import EnvironmentalMetric, EnvironmentalImpactMetrics
from .performance_metrics import PerformanceMetric, SystemPerformanceMetrics

Base = declarative_base()

class MetricsStore:
    """Persistent storage for metrics with Redis caching and PostGIS integration."""
    
    def __init__(self, 
                 base_path: Optional[Path] = None, 
                 config: Optional[MetricConfig] = None,
                 redis_url: Optional[str] = None,
                 postgres_url: Optional[str] = None):
        """Initialize metrics store with caching and database connections."""
        self.base_path = base_path or Path(__file__).parent.parent.parent / "metrics"
        self.base_path.mkdir(exist_ok=True)
        
        # Create metric type subdirectories
        self.model_metrics_path = self.base_path / "model_metrics"
        self.environmental_metrics_path = self.base_path / "environmental_metrics"
        self.performance_metrics_path = self.base_path / "performance_metrics"
        
        for path in [self.model_metrics_path, self.environmental_metrics_path,
                    self.performance_metrics_path]:
            path.mkdir(exist_ok=True)

        # Initialize Redis cache
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
        # Initialize PostgreSQL with PostGIS
        if postgres_url:
            self.engine = create_engine(postgres_url)
            Base.metadata.create_all(self.engine)
        else:
            self.engine = None

        # Store configuration
        self.config = config or MetricConfig()
        
        # Cache configuration
        self.cache_ttl = 3600  # 1 hour default TTL

    def _get_cache_key(self, metric_type: str, **kwargs) -> str:
        """Generate cache key based on metric type and filters."""
        key_parts = [metric_type]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()) if v is not None)
        return "metrics:" + ":".join(key_parts)

    def _save_metrics(self, metrics: List[BaseMetric], directory: Path):
        """Save metrics to JSON file and cache."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = directory / f"metrics_{timestamp}.json"
        
        metrics_data = [m.to_dict() for m in metrics]
        
        # Save to file
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        # Save to cache if available
        if self.redis_client:
            cache_key = self._get_cache_key(metrics[0].__class__.__name__)
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(metrics_data)
            )

    def get_metrics(self, 
                    directory: Path,
                    start_time: Optional[datetime.datetime] = None,
                    end_time: Optional[datetime.datetime] = None,
                    metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve metrics from cache or storage with optional filtering."""
        cache_key = self._get_cache_key(
            directory.name,
            start_time=start_time,
            end_time=end_time,
            metric_name=metric_name
        )
        
        # Try cache first
        if self.redis_client:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        
        # Fall back to file system
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
        
        # Cache the results
        if self.redis_client:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(metrics))
        
        return metrics

    def get_model_metrics(self, 
                         model_name: Optional[str] = None,
                         metric_type: Optional[str] = None,
                         **kwargs) -> List[Dict[str, Any]]:
        """Get ML model metrics with caching and optional filtering."""
        cache_key = self._get_cache_key(
            'model',
            model_name=model_name,
            metric_type=metric_type,
            **kwargs
        )
        
        # Try cache first
        if self.redis_client:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        
        metrics = self.get_metrics(self.model_metrics_path, **kwargs)
        
        if model_name:
            metrics = [m for m in metrics if m.get('model_name') == model_name]
        if metric_type:
            metrics = [m for m in metrics if m.get('metric_type') == metric_type]
        
        # Cache the filtered results
        if self.redis_client:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(metrics))
            
        return metrics

    def get_environmental_metrics(self, 
                                impact_type: Optional[str] = None,
                                **kwargs) -> List[Dict[str, Any]]:
        """Get environmental metrics with caching and optional filtering."""
        cache_key = self._get_cache_key('environmental', impact_type=impact_type, **kwargs)
        
        # Try cache first
        if self.redis_client:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        
        metrics = self.get_metrics(self.environmental_metrics_path, **kwargs)
        
        if impact_type:
            metrics = [m for m in metrics if m.get('impact_type') == impact_type]
        
        # Cache the filtered results
        if self.redis_client:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(metrics))
            
        return metrics