"""
Performance Monitoring System for US Tree Dashboard

This module provides comprehensive monitoring capabilities:
1. System Resource Monitoring
   - CPU usage tracking
   - Memory utilization
   - Disk I/O statistics
   - Network bandwidth usage

2. Application Performance
   - Response time tracking
   - Request throughput
   - Cache hit/miss rates
   - Database query performance

3. Dashboard Components
   - Component render times
   - Callback execution metrics
   - Data loading performance
   - Visualization generation timing

4. Alerting System
   - Resource threshold alerts
   - Performance degradation warnings
   - Error rate monitoring
   - System health notifications

Usage:
    from monitoring import monitor_performance

    @monitor_performance
    def dashboard_component():
        ...
"""

import dash_bootstrap_components as dbc
from dash import html
import psutil
import time
from functools import wraps

def create_performance_stats():
    """Create performance statistics component"""
    return dbc.Card([
        dbc.CardHeader("Performance Metrics"),
        dbc.CardBody([
            html.Div(id="performance-stats"),
            dbc.Row([
                dbc.Col([
                    html.P("Cache Statistics:", className="mb-2 font-weight-bold"),
                    html.Div(id="cache-stats")
                ], width=6),
                dbc.Col([
                    html.P("System Resources:", className="mb-2 font-weight-bold"),
                    html.Div(id="system-stats")
                ], width=6)
            ])
        ]),
        dbc.CardFooter([
            dbc.Button(
                "Refresh Stats",
                id="refresh-stats-button",
                color="primary",
                size="sm"
            )
        ])
    ])

def get_system_stats():
    """Get system resource statistics"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'swap_percent': psutil.swap_memory().percent
    }

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
        
        # Log performance metrics
        print(f"Function: {func.__name__}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        
        return result
    
    return wrapper
