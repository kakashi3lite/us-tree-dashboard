"""
Performance monitoring and optimization module for the dashboard.
"""
from datetime import datetime
from typing import Dict, Optional
import time
import psutil
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_logs = {}
        self.component_timings = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def log_request(self, request_id: str, endpoint: str):
        """Log the start of a request."""
        self.request_logs[request_id] = {
            'endpoint': endpoint,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'status': None
        }
    
    def end_request(self, request_id: str, status: int):
        """Log the end of a request."""
        if request_id in self.request_logs:
            self.request_logs[request_id]['end_time'] = time.time()
            self.request_logs[request_id]['status'] = status
            self.request_logs[request_id]['duration'] = (
                self.request_logs[request_id]['end_time'] - 
                self.request_logs[request_id]['start_time']
            )
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }
    
    def get_response_metrics(self) -> Dict:
        """Calculate response time metrics."""
        if not self.request_logs:
            return {
                'avg_response_time': 0,
                'p95_response_time': 0,
                'error_rate': 0
            }
        
        durations = [log['duration'] for log in self.request_logs.values() 
                    if log['duration'] is not None]
        error_count = sum(1 for log in self.request_logs.values() 
                         if log['status'] and log['status'] >= 400)
        
        if not durations:
            return {
                'avg_response_time': 0,
                'p95_response_time': 0,
                'error_rate': 0
            }
        
        return {
            'avg_response_time': sum(durations) / len(durations),
            'p95_response_time': sorted(durations)[int(len(durations) * 0.95)],
            'error_rate': error_count / len(self.request_logs) * 100
        }
    
    def get_component_timings(self) -> Dict:
        """Get average timing for different components."""
        return {
            component: sum(times) / len(times)
            for component, times in self.component_timings.items()
            if times
        }
    
    async def run_async(self, func, *args, **kwargs):
        """Run a function asynchronously using thread pool."""
        return await self.thread_pool.submit(func, *args, **kwargs)
    
    def monitor_component(self, component_name: str):
        """Decorator to monitor component performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if component_name not in self.component_timings:
                        self.component_timings[component_name] = []
                    self.component_timings[component_name].append(duration)
                    
                    # Keep only last 100 measurements
                    if len(self.component_timings[component_name]) > 100:
                        self.component_timings[component_name].pop(0)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in {component_name}: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def clear_old_data(self, older_than_seconds: int = 3600):
        """Clear monitoring data older than specified time."""
        current_time = time.time()
        self.request_logs = {
            req_id: log for req_id, log in self.request_logs.items()
            if log['start_time'] > current_time - older_than_seconds
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
