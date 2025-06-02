"""
Performance monitoring for the dashboard.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import time
import psutil

@dataclass
class RequestMetrics:
    start_time: float
    end_time: float
    endpoint: str
    status_code: int
    response_size: int

class DashboardMonitor:
    def __init__(self):
        self.requests: Dict[str, RequestMetrics] = {}
        self.start_time = time.time()
    
    def get_uptime(self) -> float:
        """Get dashboard uptime in seconds."""
        return time.time() - self.start_time
    
    def get_system_metrics(self) -> dict:
        """Get system resource metrics."""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu,
            'memory_used_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_used_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def start_request(self, endpoint: str):
        """Start timing a request."""
        request_id = f"{endpoint}_{datetime.now().timestamp()}"
        self.requests[request_id] = RequestMetrics(
            start_time=time.time(),
            end_time=0,
            endpoint=endpoint,
            status_code=0,
            response_size=0
        )
        return request_id
    
    def end_request(self, request_id: str, status_code: int, response_size: int):
        """End timing a request and record metrics."""
        if request_id in self.requests:
            self.requests[request_id].end_time = time.time()
            self.requests[request_id].status_code = status_code
            self.requests[request_id].response_size = response_size
    
    def get_request_metrics(self) -> dict:
        """Get request performance metrics."""
        if not self.requests:
            return {
                'avg_response_time': 0,
                'total_requests': 0,
                'success_rate': 100,
                'error_count': 0
            }
        
        total_time = 0
        error_count = 0
        completed_requests = 0
        
        for req in self.requests.values():
            if req.end_time > 0:
                completed_requests += 1
                total_time += (req.end_time - req.start_time)
                if req.status_code >= 400:
                    error_count += 1
        
        return {
            'avg_response_time': total_time / completed_requests if completed_requests > 0 else 0,
            'total_requests': len(self.requests),
            'success_rate': ((completed_requests - error_count) / completed_requests * 100) if completed_requests > 0 else 100,
            'error_count': error_count
        }
    
    def get_component_metrics(self) -> dict:
        """Get performance metrics for different components."""
        endpoints = {}
        for req in self.requests.values():
            if req.end_time > 0:
                if req.endpoint not in endpoints:
                    endpoints[req.endpoint] = {
                        'count': 0,
                        'total_time': 0,
                        'errors': 0
                    }
                
                endpoints[req.endpoint]['count'] += 1
                endpoints[req.endpoint]['total_time'] += (req.end_time - req.start_time)
                if req.status_code >= 400:
                    endpoints[req.endpoint]['errors'] += 1
        
        return {
            endpoint: {
                'avg_response_time': stats['total_time'] / stats['count'],
                'request_count': stats['count'],
                'error_rate': (stats['errors'] / stats['count'] * 100)
            }
            for endpoint, stats in endpoints.items()
        }
    
    def clear_old_metrics(self, older_than_seconds: int = 3600):
        """Clear metrics older than specified time."""
        current_time = time.time()
        self.requests = {
            k: v for k, v in self.requests.items()
            if current_time - v.start_time < older_than_seconds
        }
