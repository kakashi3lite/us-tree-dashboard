"""
Base metrics collection and monitoring functionality.
"""

from typing import Dict, List, Any, Optional, Union, Type, TypeVar
from dataclasses import dataclass, asdict, field
import datetime
import numpy as np
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseMetric')

@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    collection_interval: float = 60.0  # seconds
    storage_format: str = 'json'
    retention_days: int = 30
    batch_size: int = 100
    enable_monitoring: bool = True

@dataclass
class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    name: str
    value: Any
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    version: str = field(default='1.0.0')
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate metric value."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create metric instance from dictionary."""
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class Metric(BaseMetric):
    """Concrete implementation of base metric class."""
    
    def validate(self) -> bool:
        """Validate metric value."""
        try:
            if isinstance(self.value, (int, float)):
                return not (np.isnan(self.value) or np.isinf(self.value))
            return True
        except Exception as e:
            logger.error(f"Metric validation failed: {str(e)}")
            return False

@dataclass
class MetricSeries:
    """Time series of metrics."""
    metrics: List[BaseMetric]
    
    def get_values(self) -> List[Any]:
        """Get list of metric values."""
        return [m.value for m in self.metrics]
    
    def get_timestamps(self) -> List[datetime.datetime]:
        """Get list of timestamps."""
        return [m.timestamp for m in self.metrics]
    
    def calculate_stats(self) -> Dict[str, float]:
        """Calculate basic statistics for numeric metrics."""
        values = np.array(self.get_values())
        if not len(values) or not isinstance(values[0], (int, float)):
            return {}
            
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values),
            'last_value': float(values[-1]) if len(values) > 0 else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert series to dictionary format."""
        return {
            'metrics': [m.to_dict() for m in self.metrics],
            'stats': self.calculate_stats()
        }
