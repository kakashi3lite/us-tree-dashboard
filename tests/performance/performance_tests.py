#!/usr/bin/env python3
"""
Performance testing suite for conservation dashboard.
Optimized for Eli's workflow with automated benchmarking and bottleneck detection.
"""

import time
import psutil
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
import requests
from locust import HttpUser, task, between
import memory_profiler
import cProfile
import io
import pstats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_size_mb: float
    records_processed: int
    throughput_records_per_sec: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class ConservationDashboardLoadTest(HttpUser):
    """
    Locust load testing for conservation dashboard.
    Simulates field team accessing dashboard under various conditions.
    """
    
    wait_time = between(1, 3)  # Realistic user behavior
    
    def on_start(self):
        """Setup for each user session"""
        self.regions = ['California', 'Texas', 'Florida', 'New York', 'Washington']
        self.report_types = ['conservation', 'field_summary', 'executive']
    
    @task(3)
    def view_dashboard(self):
        """Test main dashboard loading - most common operation"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200 and "Tree Dashboard" in response.text:
                response.success()
            else:
                response.failure(f"Dashboard failed to load: {response.status_code}")
    
    @task(2)
    def filter_by_state(self):
        """Test state filtering functionality"""
        state = np.random.choice(self.regions)
        with self.client.get(f"/?state={state}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"State filtering failed: {response.status_code}")
    
    @task(1)
    def api_health_check(self):
        """Test API health endpoint"""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("API reported unhealthy status")
            else:
                response.failure(f"API health check failed: {response.status_code}")
    
    @task(1)
    def get_conservation_data(self):
        """Test conservation data API endpoint"""
        with self.client.get("/api/v1/datasets/conservation_status", 
                           catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if len(data) > 0:
                        response.success()
                    else:
                        response.failure("No conservation data returned")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Conservation data API failed: {response.status_code}")

class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for conservation data processing.
    Measures performance across various data sizes and operations.
    """
    
    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[PerformanceMetrics] = []
        
    def measure_performance(self, func, *args, **kwargs) -> PerformanceMetrics:
        """Decorator to measure function performance"""
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        success = True
        error_message = None
        result = None
        
        try:
            # Execute function
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Performance test failed: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get final system state
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = process.cpu_percent()
        
        # Calculate metrics
        memory_usage = max(final_memory - initial_memory, 0)
        cpu_usage = (initial_cpu + final_cpu) / 2
        
        # Estimate data size and throughput
        data_size = self._estimate_data_size(result)
        records_processed = self._count_records(result)
        throughput = records_processed / duration if duration > 0 else 0
        
        metrics = PerformanceMetrics(
            operation=func.__name__,
            duration_seconds=duration,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            data_size_mb=data_size,
            records_processed=records_processed,
            throughput_records_per_sec=throughput,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def _estimate_data_size(self, data) -> float:
        """Estimate data size in MB"""
        if data is None:
            return 0.0
        
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum() / 1024 / 1024
        elif isinstance(data, (list, dict)):
            return len(str(data)) / 1024 / 1024
        else:
            return 0.1  # Default small size
    
    def _count_records(self, data) -> int:
        """Count number of records processed"""
        if data is None:
            return 0
        
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data)
        else:
            return 1

    @pytest.mark.benchmark(group="data_loading")
    def test_gbif_data_loading(self, benchmark):
        """Benchmark GBIF data loading performance"""
        def load_gbif_data():
            # Simulate loading large GBIF dataset
            data_size = np.random.choice([1000, 5000, 10000, 50000])
            return pd.DataFrame({
                'scientificName': np.random.choice(['Species A', 'Species B', 'Species C'], data_size),
                'latitude': np.random.uniform(25, 50, data_size),
                'longitude': np.random.uniform(-125, -65, data_size),
                'year': np.random.randint(2000, 2024, data_size)
            })
        
        result = benchmark(load_gbif_data)
        self.measure_performance(load_gbif_data)
        return result
    
    @pytest.mark.benchmark(group="data_processing")
    def test_canopy_analysis(self, benchmark):
        """Benchmark canopy coverage analysis"""
        # Create test canopy data
        test_data = pd.DataFrame({
            'state': np.random.choice(['CA', 'TX', 'FL', 'NY'], 1000),
            'county': [f'County_{i}' for i in range(1000)],
            'canopy_pct': np.random.uniform(10, 80, 1000),
            'area_km2': np.random.uniform(50, 500, 1000)
        })
        
        def analyze_canopy_coverage(data):
            # Simulate complex canopy analysis
            state_stats = data.groupby('state').agg({
                'canopy_pct': ['mean', 'std', 'min', 'max'],
                'area_km2': 'sum'
            })
            
            # Calculate weighted averages
            weighted_avg = (data['canopy_pct'] * data['area_km2']).sum() / data['area_km2'].sum()
            
            return {
                'state_stats': state_stats,
                'weighted_average': weighted_avg,
                'total_area': data['area_km2'].sum()
            }
        
        result = benchmark(analyze_canopy_coverage, test_data)
        self.measure_performance(analyze_canopy_coverage, test_data)
        return result
    
    @pytest.mark.benchmark(group="visualization")
    def test_chart_generation(self, benchmark):
        """Benchmark chart generation performance"""
        import plotly.express as px
        
        # Create test data for visualization
        test_data = pd.DataFrame({
            'species': np.random.choice(['Oak', 'Maple', 'Pine', 'Birch'], 500),
            'health': np.random.choice(['Good', 'Fair', 'Poor'], 500),
            'dbh': np.random.uniform(10, 100, 500),
            'height': np.random.uniform(20, 150, 500)
        })
        
        def generate_charts(data):
            # Generate multiple chart types
            charts = {}
            
            # Species distribution chart
            species_counts = data['species'].value_counts()
            charts['species'] = px.bar(x=species_counts.index, y=species_counts.values)
            
            # Health distribution chart
            health_counts = data['health'].value_counts()
            charts['health'] = px.pie(values=health_counts.values, names=health_counts.index)
            
            # Scatter plot
            charts['scatter'] = px.scatter(data, x='dbh', y='height', color='health')
            
            return charts
        
        result = benchmark(generate_charts, test_data)
        self.measure_performance(generate_charts, test_data)
        return result
    
    def test_concurrent_users(self, max_users: int = 50):
        """Test performance under concurrent user load"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def simulate_user_session():
            """Simulate a user session"""
            start_time = time.time()
            try:
                # Simulate user actions
                time.sleep(np.random.uniform(0.1, 0.5))  # Page load time
                response = requests.get("http://localhost:8050/", timeout=10)
                
                # Simulate filtering
                time.sleep(np.random.uniform(0.1, 0.3))
                state = np.random.choice(['CA', 'TX', 'FL', 'NY'])
                response = requests.get(f"http://localhost:8050/?state={state}", timeout=10)
                
                duration = time.time() - start_time
                results_queue.put({
                    'success': response.status_code == 200,
                    'duration': duration,
                    'status_code': response.status_code
                })
                
            except Exception as e:
                results_queue.put({
                    'success': False,
                    'duration': time.time() - start_time,
                    'error': str(e)
                })
        
        # Run concurrent user sessions
        threads = []
        for i in range(max_users):
            thread = threading.Thread(target=simulate_user_session)
            threads.append(thread)
            thread.start()
            time.sleep(0.1)  # Stagger start times
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Analyze results
        successful_requests = sum(1 for r in results if r['success'])
        avg_duration = np.mean([r['duration'] for r in results])
        max_duration = np.max([r['duration'] for r in results])
        
        metrics = {
            'concurrent_users': max_users,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / len(results),
            'avg_response_time': avg_duration,
            'max_response_time': max_duration,
            'total_requests': len(results)
        }
        
        logger.info(f"Concurrent user test results: {metrics}")
        return metrics
    
    def test_memory_leaks(self, iterations: int = 100):
        """Test for memory leaks during repeated operations"""
        import gc
        
        memory_usage = []
        
        for i in range(iterations):
            # Simulate data loading and processing
            data = pd.DataFrame({
                'col1': np.random.randn(1000),
                'col2': np.random.randn(1000),
                'col3': np.random.choice(['A', 'B', 'C'], 1000)
            })
            
            # Process data
            result = data.groupby('col3').sum()
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append(memory_mb)
            
            if i % 10 == 0:
                logger.info(f"Iteration {i}: Memory usage {memory_mb:.1f} MB")
        
        # Analyze memory trend
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        
        metrics = {
            'iterations': iterations,
            'initial_memory_mb': memory_usage[0],
            'final_memory_mb': memory_usage[-1],
            'max_memory_mb': max(memory_usage),
            'memory_trend_mb_per_iteration': memory_trend,
            'potential_leak': memory_trend > 0.1  # > 0.1 MB per iteration
        }
        
        logger.info(f"Memory leak test results: {metrics}")
        return metrics
    
    def profile_critical_functions(self):
        """Profile critical conservation dashboard functions"""
        # Import dashboard functions
        try:
            from app import TreeDashboard
            dashboard = TreeDashboard()
        except ImportError:
            logger.warning("Could not import TreeDashboard for profiling")
            return
        
        # Profile data loading
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            dashboard.load_datasets()
        except:
            pass
        
        profiler.disable()
        
        # Save profiling results
        profile_output = io.StringIO()
        stats = pstats.Stats(profiler, stream=profile_output)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        # Save to file
        profile_file = self.output_dir / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(profile_file, 'w') as f:
            f.write(profile_output.getvalue())
        
        logger.info(f"Profiling results saved to {profile_file}")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Calculate summary statistics
        if not self.metrics:
            logger.warning("No performance metrics available")
            return str(report_file)
        
        avg_duration = np.mean([m.duration_seconds for m in self.metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in self.metrics])
        avg_throughput = np.mean([m.throughput_records_per_sec for m in self.metrics if m.throughput_records_per_sec > 0])
        success_rate = sum(1 for m in self.metrics if m.success) / len(self.metrics)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conservation Dashboard Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #28a745; color: white; padding: 20px; text-align: center; }}
                .metric {{ display: inline-block; margin: 20px; text-align: center; }}
                .metric h3 {{ color: #28a745; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Performance Testing Report</h1>
                <p>Conservation Dashboard Performance Analysis</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metric">
                <h3>Average Duration</h3>
                <p>{avg_duration:.2f} seconds</p>
            </div>
            <div class="metric">
                <h3>Average Memory Usage</h3>
                <p>{avg_memory:.1f} MB</p>
            </div>
            <div class="metric">
                <h3>Average Throughput</h3>
                <p>{avg_throughput:.0f} records/sec</p>
            </div>
            <div class="metric">
                <h3>Success Rate</h3>
                <p>{success_rate:.1%}</p>
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Operation</th>
                    <th>Duration (s)</th>
                    <th>Memory (MB)</th>
                    <th>CPU (%)</th>
                    <th>Records</th>
                    <th>Throughput</th>
                    <th>Status</th>
                </tr>
        """
        
        for metric in self.metrics:
            status_class = "success" if metric.success else "failure"
            status_text = "✅ Success" if metric.success else f"❌ Failed: {metric.error_message}"
            
            html_content += f"""
                <tr>
                    <td>{metric.operation}</td>
                    <td>{metric.duration_seconds:.2f}</td>
                    <td>{metric.memory_usage_mb:.1f}</td>
                    <td>{metric.cpu_usage_percent:.1f}</td>
                    <td>{metric.records_processed:,}</td>
                    <td>{metric.throughput_records_per_sec:.0f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Performance Recommendations</h2>
            <ul>
        """
        
        # Generate recommendations based on results
        if avg_duration > 5:
            html_content += "<li>⚠️ Consider optimizing slow operations (>5s average duration)</li>"
        
        if avg_memory > 500:
            html_content += "<li>⚠️ High memory usage detected (>500MB), consider data streaming</li>"
        
        if success_rate < 0.95:
            html_content += "<li>❌ Low success rate (<95%), investigate error causes</li>"
        
        if avg_throughput < 100:
            html_content += "<li>⚡ Low throughput (<100 records/sec), optimize data processing</li>"
        
        html_content += """
                <li>✅ Continue monitoring performance metrics in production</li>
                <li>✅ Set up automated performance regression testing</li>
            </ul>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {report_file}")
        return str(report_file)

def run_full_performance_suite():
    """Run complete performance testing suite"""
    benchmark = PerformanceBenchmark()
    
    logger.info("🚀 Starting conservation dashboard performance testing suite")
    
    # Run individual benchmarks
    logger.info("Testing data loading performance...")
    benchmark.test_gbif_data_loading(lambda f: f())
    
    logger.info("Testing data processing performance...")
    benchmark.test_canopy_analysis(lambda f, d: f(d))
    
    logger.info("Testing visualization performance...")
    benchmark.test_chart_generation(lambda f, d: f(d))
    
    # Test concurrent users (if server is running)
    try:
        logger.info("Testing concurrent user performance...")
        benchmark.test_concurrent_users(max_users=20)
    except Exception as e:
        logger.warning(f"Concurrent user testing failed: {e}")
    
    # Test memory leaks
    logger.info("Testing for memory leaks...")
    benchmark.test_memory_leaks(iterations=50)
    
    # Profile critical functions
    logger.info("Profiling critical functions...")
    benchmark.profile_critical_functions()
    
    # Generate report
    logger.info("Generating performance report...")
    report_path = benchmark.generate_performance_report()
    
    logger.info(f"✅ Performance testing complete! Report: {report_path}")
    return report_path

if __name__ == "__main__":
    run_full_performance_suite()
