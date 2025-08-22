"""
Metrics collection for machine learning models.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
import numpy as np
from dataclasses import dataclass, field
import psutil
import time
from .base import BaseMetric, MetricSeries

@dataclass
class ModelMetric(BaseMetric):
    """Metric specific to ML model performance.

    Defaults provided to avoid ordering conflict with BaseMetric defaults.
    """
    model_name: str = ""
    metric_type: str = ""
    confidence: Optional[float] = None
    prediction_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate model metric value."""
        if self.metric_type in ['accuracy', 'f1_score', 'r2_score']:
            return isinstance(self.value, (int, float)) and 0 <= self.value <= 1
        elif self.metric_type in ['rmse', 'mae', 'latency']:
            return isinstance(self.value, (int, float)) and self.value >= 0
        return super().validate()

@dataclass
class ModelTrainingMetrics:
    """Collection of metrics from a model training session.

    Note: All required (non-default) fields must appear before any with defaults
    to satisfy dataclass ordering constraints.
    """
    model_name: str
    version: str
    timestamp: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    dataset_size: int
    training_time: float
    feature_importance: Dict[str, float] | None = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    training_start: datetime.datetime = field(default_factory=datetime.datetime.now)
    error_rate: Optional[float] = None

    def __post_init__(self):
        """Initialize resource usage tracking."""
        self.resource_usage = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_used': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'available_memory': psutil.virtual_memory().available / 1024 / 1024  # MB
        }

    def to_metric_list(self) -> List[ModelMetric]:
        """Convert training metrics to list of individual metrics."""
        base_tags = {
            'model_name': self.model_name,
            'version': self.version
        }

        metrics = []
        timestamp = datetime.datetime.now()

        # Performance metrics
        for name, value in self.metrics.items():
            metrics.append(ModelMetric(
                name=f"{self.model_name}_{name}",
                value=value,
                timestamp=timestamp,
                tags=base_tags.copy(),
                model_name=self.model_name,
                metric_type='performance'
            ))

        # Feature importance metrics
        if self.feature_importance:
            for feature, importance in self.feature_importance.items():
                metrics.append(ModelMetric(
                    name=f"{self.model_name}_feature_importance_{feature}",
                    value=importance,
                    timestamp=timestamp,
                    tags=base_tags.copy(),
                    model_name=self.model_name,
                    metric_type='feature_importance'
                ))

        # Resource usage metrics
        for resource, value in self.resource_usage.items():
            metrics.append(ModelMetric(
                name=f"{self.model_name}_resource_{resource}",
                value=value,
                timestamp=timestamp,
                tags=base_tags.copy(),
                model_name=self.model_name,
                metric_type='resource'
            ))

        # Training time
        metrics.append(ModelMetric(
            name=f"{self.model_name}_training_time",
            value=self.training_time,
            timestamp=timestamp,
            tags=base_tags.copy(),
            model_name=self.model_name,
            metric_type='timing'
        ))

        if self.error_rate is not None:
            metrics.append(ModelMetric(
                name=f"{self.model_name}_error_rate",
                value=self.error_rate,
                timestamp=timestamp,
                tags=base_tags.copy(),
                model_name=self.model_name,
                metric_type='error'
            ))

        return metrics

@dataclass
class ModelPredictionMetrics:
    """Metrics for model predictions."""
    model_name: str
    prediction_id: str
    prediction_time: float
    confidence_scores: Dict[str, float]
    latency: float
    error: Optional[str] = None
    
    def to_metric_list(self) -> List[ModelMetric]:
        """Convert prediction metrics to list of metrics."""
        base_tags = {
            'model_name': self.model_name,
            'prediction_id': self.prediction_id
        }
        
        metrics = []
        timestamp = datetime.datetime.now()
        
        metrics.append(ModelMetric(
            name=f"{self.model_name}_prediction_time",
            value=self.prediction_time,
            timestamp=timestamp,
            tags=base_tags.copy(),
            model_name=self.model_name,
            metric_type='prediction',
            prediction_id=self.prediction_id
        ))
        
        metrics.append(ModelMetric(
            name=f"{self.model_name}_prediction_latency",
            value=self.latency,
            timestamp=timestamp,
            tags=base_tags.copy(),
            model_name=self.model_name,
            metric_type='latency',
            prediction_id=self.prediction_id
        ))
        
        # Average confidence score
        if self.confidence_scores:
            avg_confidence = np.mean(list(self.confidence_scores.values()))
            metrics.append(ModelMetric(
                name=f"{self.model_name}_prediction_confidence",
                value=avg_confidence,
                timestamp=timestamp,
                tags=base_tags.copy(),
                model_name=self.model_name,
                metric_type='confidence',
                prediction_id=self.prediction_id,
                confidence=avg_confidence
            ))
        
        return metrics
                
def collect_prediction_metrics(model_name: str,
                             start_time: float,
                             confidence_scores: Dict[str, float],
                             prediction_id: Optional[str] = None) -> ModelPredictionMetrics:
    """Collect metrics for a prediction."""
    end_time = time.time()
    latency = end_time - start_time
    
    return ModelPredictionMetrics(
        model_name=model_name,
        prediction_id=prediction_id or str(int(start_time * 1000)),
        prediction_time=end_time,
        confidence_scores=confidence_scores,
        latency=latency
    )
