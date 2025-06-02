"""
Enhanced performance monitoring for the dashboard.
"""
from datetime import datetime
import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

@dataclass
class EnhancedRequestMetrics:
    start_time: float
    end_time: float = 0
    endpoint: str = ""
    status_code: int = 0
    response_size: int = 0
    cache_hit: bool = False
    memory_used: float = 0
    cpu_usage: float = 0

class EnhancedDashboardMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.requests: Dict[str, EnhancedRequestMetrics] = {}
        self.component_timings: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.response_time_window: List[float] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.last_cleanup = time.time()
    
    def log_request(self, endpoint: str, cache_hit: bool = False) -> str:
        """Start logging a new request with enhanced metrics."""
        request_id = f"{endpoint}_{datetime.now().timestamp()}"
        self.requests[request_id] = EnhancedRequestMetrics(
            start_time=time.time(),
            endpoint=endpoint,
            cache_hit=cache_hit,
            memory_used=psutil.Process().memory_info().rss / (1024 * 1024),  # MB
            cpu_usage=psutil.Process().cpu_percent()
        )
        return request_id
    
    def end_request(self, request_id: str, status_code: int, response_size: int):
        """Complete request logging with final metrics."""
        if request_id in self.requests:
            req = self.requests[request_id]
            req.end_time = time.time()
            req.status_code = status_code
            req.response_size = response_size
            duration = req.end_time - req.start_time
            
            # Update response time window
            self.response_time_window.append(duration)
            if len(self.response_time_window) > 1000:
                self.response_time_window.pop(0)
            
            # Update cache stats
            if req.cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def monitor_memory_persistence(self):
        """Monitor memory persistence operations."""
        from mcp_server_config.mindmap_handlers import mindmap_handler
        
        try:
            stats = {
                'nodes': len(mindmap_handler.cache),
                'backups': len(list(Path(mindmap_handler.data_dir / 'backups').glob('*/'))),
                'last_backup': None,
                'disk_usage': None
            }
            
            # Get latest backup
            backup_dirs = sorted(Path(mindmap_handler.data_dir / 'backups').glob('*/'))
            if backup_dirs:
                stats['last_backup'] = backup_dirs[-1].name
            
            # Calculate disk usage
            total_size = 0
            for file in mindmap_handler.data_dir.rglob('*.json'):
                total_size += file.stat().st_size
            stats['disk_usage'] = total_size / (1024 * 1024)  # MB
            
            return stats
        except Exception as e:
            print(f"Error monitoring memory persistence: {e}")
            return None

    def get_enhanced_metrics(self) -> dict:
        """Get comprehensive performance metrics."""
        current_time = time.time()
        
        # Filter recent requests (last hour)
        recent_requests = {
            req_id: req for req_id, req in self.requests.items()
            if req.start_time > current_time - 3600
        }
        
        if not recent_requests:
            return self._get_empty_metrics()
        
        completed_requests = [req for req in recent_requests.values() if req.end_time > 0]
        if not completed_requests:
            return self._get_empty_metrics()
        
        # Calculate response time statistics
        response_times = [req.end_time - req.start_time for req in completed_requests]
        response_times = np.array(response_times)
        
        # Calculate error rate
        error_count = sum(1 for req in completed_requests if req.status_code >= 400)
        error_rate = (error_count / len(completed_requests)) * 100
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        metrics = {
            'response_time': {
                'average': float(np.mean(response_times)),
                'median': float(np.median(response_times)),
                'p95': float(np.percentile(response_times, 95)),
                'p99': float(np.percentile(response_times, 99))
            },
            'requests': {
                'total': len(recent_requests),
                'completed': len(completed_requests),
                'error_rate': error_rate
            },
            'cache': {
                'hit_rate': cache_hit_rate,
                'hits': self.cache_hits,
                'misses': self.cache_misses
            },
            'system': system_metrics,
            'uptime': current_time - self.start_time
        }
        
        memory_stats = self.monitor_memory_persistence()
        if memory_stats:
            metrics['memory_persistence'] = memory_stats
        
        return metrics
    
    def _get_system_metrics(self) -> dict:
        """Get detailed system resource metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'usage': psutil.cpu_percent(interval=0.1),
                'cores': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'used_percent': memory.percent
            },
            'disk': {
                'total': disk.total / (1024**3),  # GB
                'free': disk.free / (1024**3),  # GB
                'used_percent': disk.percent
            }
        }
    
    def _get_empty_metrics(self) -> dict:
        """Return empty metrics structure."""
        return {
            'response_time': {
                'average': 0,
                'median': 0,
                'p95': 0,
                'p99': 0
            },
            'requests': {
                'total': 0,
                'completed': 0,
                'error_rate': 0
            },
            'cache': {
                'hit_rate': 0,
                'hits': 0,
                'misses': 0
            },
            'system': self._get_system_metrics(),
            'uptime': time.time() - self.start_time
        }
    
    def monitor_component(self, component_name: str):
        """Decorator for monitoring component performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if component_name not in self.component_timings:
                    self.component_timings[component_name] = []
                    
                self.component_timings[component_name].append(duration)
                if len(self.component_timings[component_name]) > 100:
                    self.component_timings[component_name].pop(0)
                
                return result
            return wrapper
        return decorator
    
    def cleanup_old_data(self):
        """Clean up old monitoring data periodically."""
        current_time = time.time()
        if current_time - self.last_cleanup > 3600:  # Cleanup every hour
            self.requests = {
                req_id: req for req_id, req in self.requests.items()
                if current_time - req.start_time < 86400  # Keep last 24 hours
            }
            self.last_cleanup = current_time

# Global instance
enhanced_monitor = EnhancedDashboardMonitor()
