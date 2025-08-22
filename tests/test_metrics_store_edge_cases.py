"""
Unit tests for metrics store functionality with mocked dependencies.

These tests focus on key metrics store functions while mocking heavy dependencies
to improve coverage without requiring pandas/numpy.
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics_store_stub import MetricsStore, BaseMetric, MetricConfig


class TestMetricsStoreEdgeCases(unittest.TestCase):
    """Test edge cases for metrics store functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.metrics_store = MetricsStore(base_path=Path(self.test_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_metrics_store_initialization(self):
        """Test metrics store initialization with various parameters."""
        # Test with default parameters
        default_store = MetricsStore()
        self.assertIsNotNone(default_store.base_path)
        self.assertIsInstance(default_store.config, MetricConfig)
        
        # Test with custom config
        custom_config = MetricConfig(retention_days=60, max_metrics_per_file=500)
        custom_store = MetricsStore(config=custom_config)
        self.assertEqual(custom_store.config.retention_days, 60)
        self.assertEqual(custom_store.config.max_metrics_per_file, 500)
        
        # Test that directories are created
        self.assertTrue(self.metrics_store.model_metrics_path.exists())
        self.assertTrue(self.metrics_store.environmental_metrics_path.exists())
        self.assertTrue(self.metrics_store.performance_metrics_path.exists())
    
    def test_base_metric_creation_and_serialization(self):
        """Test BaseMetric creation and serialization."""
        test_metric = BaseMetric(
            name="test_metric",
            value=0.85,
            timestamp=datetime.now(),
            tags={"model": "test_model", "version": "1.0"}
        )
        
        # Test to_dict conversion
        metric_dict = test_metric.to_dict()
        self.assertIsInstance(metric_dict, dict)
        self.assertEqual(metric_dict["name"], "test_metric")
        self.assertEqual(metric_dict["value"], 0.85)
        self.assertIn("timestamp", metric_dict)
        self.assertEqual(metric_dict["tags"]["model"], "test_model")
    
    def test_save_model_metrics_edge_cases(self):
        """Test saving model metrics with various input types."""
        # Test with BaseMetric object
        metric_obj = BaseMetric("accuracy", 0.95, datetime.now())
        self.metrics_store.save_model_metrics(metric_obj)
        
        # Test with dictionary input
        metric_dict = {
            "name": "precision",
            "value": 0.88,
            "tags": {"model_type": "classifier"}
        }
        self.metrics_store.save_model_metrics(metric_dict)
        
        # Test with list of metrics
        metric_list = [
            BaseMetric("recall", 0.92, datetime.now()),
            BaseMetric("f1_score", 0.90, datetime.now())
        ]
        self.metrics_store.save_model_metrics(metric_list)
        
        # Verify files were created
        metric_files = list(self.metrics_store.model_metrics_path.glob("metrics_*.json"))
        self.assertGreater(len(metric_files), 0)
    
    def test_save_environmental_metrics(self):
        """Test saving environmental metrics."""
        env_metric = {
            "name": "carbon_footprint", 
            "value": 125.5,
            "tags": {"source": "data_processing", "units": "kg_co2"}
        }
        
        self.metrics_store.save_environmental_metrics(env_metric)
        
        # Verify file was created
        env_files = list(self.metrics_store.environmental_metrics_path.glob("metrics_*.json"))
        self.assertGreater(len(env_files), 0)
    
    def test_save_performance_metrics(self):
        """Test saving performance metrics."""
        perf_metric = {
            "name": "response_time",
            "value": 250,
            "tags": {"endpoint": "api/data", "units": "ms"}
        }
        
        self.metrics_store.save_performance_metrics(perf_metric)
        
        # Verify file was created
        perf_files = list(self.metrics_store.performance_metrics_path.glob("metrics_*.json"))
        self.assertGreater(len(perf_files), 0)
    
    def test_get_metrics_with_filtering(self):
        """Test retrieving metrics with various filters."""
        # Save test metrics with different timestamps
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        
        test_metrics = [
            BaseMetric("old_metric", 100, old_time),
            BaseMetric("new_metric", 200, now),
            BaseMetric("another_metric", 300, now)
        ]
        
        self.metrics_store.save_model_metrics(test_metrics)
        
        # Test getting all metrics
        all_metrics = self.metrics_store.get_metrics(self.metrics_store.model_metrics_path)
        self.assertGreaterEqual(len(all_metrics), 3)
        
        # Test filtering by time range
        recent_start = now - timedelta(minutes=30)
        recent_metrics = self.metrics_store.get_metrics(
            self.metrics_store.model_metrics_path,
            start_time=recent_start
        )
        
        # Should exclude the old metric
        metric_names = [m["name"] for m in recent_metrics]
        self.assertIn("new_metric", metric_names)
        self.assertIn("another_metric", metric_names)
        
        # Test filtering by metric name
        specific_metrics = self.metrics_store.get_metrics(
            self.metrics_store.model_metrics_path,
            metric_name="new_metric"
        )
        self.assertEqual(len(specific_metrics), 1)
        self.assertEqual(specific_metrics[0]["name"], "new_metric")
    
    def test_get_model_metrics_with_filters(self):
        """Test get_model_metrics with model name and type filters."""
        # Save metrics with tags
        metrics_with_tags = [
            BaseMetric("accuracy", 0.95, datetime.now(), tags={"model_name": "tree_classifier", "metric_type": "accuracy"}),
            BaseMetric("precision", 0.88, datetime.now(), tags={"model_name": "tree_classifier", "metric_type": "precision"}),
            BaseMetric("accuracy", 0.92, datetime.now(), tags={"model_name": "species_classifier", "metric_type": "accuracy"})
        ]
        
        self.metrics_store.save_model_metrics(metrics_with_tags)
        
        # Test filtering by model name
        tree_metrics = self.metrics_store.get_model_metrics(model_name="tree_classifier")
        self.assertEqual(len(tree_metrics), 2)
        
        # Test filtering by metric type
        accuracy_metrics = self.metrics_store.get_model_metrics(metric_type="accuracy")
        self.assertEqual(len(accuracy_metrics), 2)
        
        # Test filtering by both
        specific_metrics = self.metrics_store.get_model_metrics(
            model_name="tree_classifier", 
            metric_type="precision"
        )
        self.assertEqual(len(specific_metrics), 1)
        self.assertEqual(specific_metrics[0]["name"], "precision")
    
    def test_get_environmental_metrics_with_filters(self):
        """Test get_environmental_metrics with impact type filter."""
        env_metrics = [
            BaseMetric("energy_usage", 150, datetime.now(), tags={"impact_type": "carbon"}),
            BaseMetric("water_usage", 200, datetime.now(), tags={"impact_type": "water"}),
            BaseMetric("carbon_emissions", 75, datetime.now(), tags={"impact_type": "carbon"})
        ]
        
        self.metrics_store.save_environmental_metrics(env_metrics)
        
        # Test filtering by impact type
        carbon_metrics = self.metrics_store.get_environmental_metrics(impact_type="carbon")
        self.assertEqual(len(carbon_metrics), 2)
        
        water_metrics = self.metrics_store.get_environmental_metrics(impact_type="water")
        self.assertEqual(len(water_metrics), 1)
    
    def test_get_performance_metrics_with_filters(self):
        """Test get_performance_metrics with component and type filters."""
        perf_metrics = [
            BaseMetric("cpu_usage", 45, datetime.now(), tags={"component": "api", "metric_type": "resource"}),
            BaseMetric("memory_usage", 60, datetime.now(), tags={"component": "api", "metric_type": "resource"}),
            BaseMetric("response_time", 120, datetime.now(), tags={"component": "database", "metric_type": "latency"})
        ]
        
        self.metrics_store.save_performance_metrics(perf_metrics)
        
        # Test filtering by component
        api_metrics = self.metrics_store.get_performance_metrics(component="api")
        self.assertEqual(len(api_metrics), 2)
        
        # Test filtering by metric type
        resource_metrics = self.metrics_store.get_performance_metrics(metric_type="resource")
        self.assertEqual(len(resource_metrics), 2)
        
        # Test filtering by both
        api_latency = self.metrics_store.get_performance_metrics(
            component="database", 
            metric_type="latency"
        )
        self.assertEqual(len(api_latency), 1)
    
    def test_metric_trends_calculation(self):
        """Test trend calculation for metrics."""
        # Create metrics with trend data
        base_time = datetime.now()
        trending_metrics = [
            {"name": "increasing_metric", "value": 10, "timestamp": (base_time - timedelta(hours=3)).isoformat()},
            {"name": "increasing_metric", "value": 15, "timestamp": (base_time - timedelta(hours=2)).isoformat()},
            {"name": "increasing_metric", "value": 20, "timestamp": (base_time - timedelta(hours=1)).isoformat()},
            
            {"name": "decreasing_metric", "value": 100, "timestamp": (base_time - timedelta(hours=3)).isoformat()},
            {"name": "decreasing_metric", "value": 80, "timestamp": (base_time - timedelta(hours=2)).isoformat()},
            {"name": "decreasing_metric", "value": 60, "timestamp": (base_time - timedelta(hours=1)).isoformat()},
            
            {"name": "stable_metric", "value": 50, "timestamp": (base_time - timedelta(hours=2)).isoformat()},
            {"name": "stable_metric", "value": 51, "timestamp": (base_time - timedelta(hours=1)).isoformat()},
            
            {"name": "single_point", "value": 25, "timestamp": base_time.isoformat()}
        ]
        
        trends = self.metrics_store.get_metric_trends(trending_metrics)
        
        # Test increasing trend
        self.assertEqual(trends["increasing_metric"]["trend"], "increasing")
        self.assertGreater(trends["increasing_metric"]["change"], 5)
        
        # Test decreasing trend
        self.assertEqual(trends["decreasing_metric"]["trend"], "decreasing")
        self.assertLess(trends["decreasing_metric"]["change"], -5)
        
        # Test stable trend
        self.assertEqual(trends["stable_metric"]["trend"], "stable")
        
        # Test insufficient data
        self.assertEqual(trends["single_point"]["trend"], "insufficient_data")
    
    def test_export_metrics_report(self):
        """Test metrics report export functionality."""
        # Add various metrics
        self.metrics_store.save_model_metrics(BaseMetric("test_accuracy", 0.95, datetime.now()))
        self.metrics_store.save_environmental_metrics(BaseMetric("carbon_footprint", 100, datetime.now()))
        self.metrics_store.save_performance_metrics(BaseMetric("cpu_usage", 45, datetime.now()))
        
        # Export report
        report_path = Path(self.test_dir) / "test_report.json"
        self.metrics_store.export_metrics_report(report_path)
        
        # Verify report was created and has expected structure
        self.assertTrue(report_path.exists())
        
        with open(report_path) as f:
            report = json.load(f)
        
        self.assertIn("generated_at", report)
        self.assertIn("summary", report)
        self.assertIn("model_metrics", report)
        self.assertIn("environmental_metrics", report)
        self.assertIn("performance_metrics", report)
        self.assertIn("trends", report)
        
        # Check summary counts
        self.assertGreaterEqual(report["summary"]["total_model_metrics"], 1)
        self.assertGreaterEqual(report["summary"]["total_environmental_metrics"], 1)
        self.assertGreaterEqual(report["summary"]["total_performance_metrics"], 1)
    
    def test_cleanup_old_metrics(self):
        """Test cleanup of old metrics files."""
        # Create old metric files by manually creating files with old timestamps
        old_timestamp = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d_%H%M%S")
        recent_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        old_file = self.metrics_store.model_metrics_path / f"metrics_{old_timestamp}.json"
        recent_file = self.metrics_store.model_metrics_path / f"metrics_{recent_timestamp}.json"
        
        # Create files
        with open(old_file, 'w') as f:
            json.dump([{"name": "old_metric", "value": 1}], f)
        
        with open(recent_file, 'w') as f:
            json.dump([{"name": "recent_metric", "value": 2}], f)
        
        # Run cleanup with 30 day retention
        self.metrics_store.cleanup_old_metrics(days=30)
        
        # Check that old file was removed and recent file remains
        self.assertFalse(old_file.exists())
        self.assertTrue(recent_file.exists())
    
    def test_error_handling_in_file_operations(self):
        """Test error handling during file operations."""
        # Test saving to read-only directory (simulate permission error)
        # This is a bit tricky to test portably, so we'll test with invalid paths
        
        # Test get_metrics with non-existent directory
        non_existent_dir = Path("/path/that/does/not/exist")
        metrics = self.metrics_store.get_metrics(non_existent_dir)
        self.assertEqual(len(metrics), 0)
        
        # Test reading corrupted JSON file
        corrupt_file = self.metrics_store.model_metrics_path / "metrics_corrupt.json"
        with open(corrupt_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should handle corrupted file gracefully
        metrics = self.metrics_store.get_metrics(self.metrics_store.model_metrics_path)
        # Should return metrics from valid files, ignore corrupted one
        self.assertIsInstance(metrics, list)
    
    def test_edge_case_metric_values(self):
        """Test handling of edge case metric values."""
        edge_case_metrics = [
            BaseMetric("zero_value", 0, datetime.now()),
            BaseMetric("negative_value", -10.5, datetime.now()),
            BaseMetric("large_value", 1e10, datetime.now()),
            BaseMetric("float_precision", 0.12345678901234567890, datetime.now()),
            BaseMetric("string_value", "text_metric", datetime.now())
        ]
        
        # Should handle all edge cases without error
        try:
            self.metrics_store.save_model_metrics(edge_case_metrics)
            retrieved_metrics = self.metrics_store.get_model_metrics()
            self.assertGreaterEqual(len(retrieved_metrics), 5)
        except Exception as e:
            self.fail(f"Should handle edge case values gracefully: {e}")
    
    def test_concurrent_file_access_simulation(self):
        """Test behavior with concurrent-like file access."""
        # Simulate concurrent writes by rapidly saving metrics
        for i in range(10):
            metric = BaseMetric(f"concurrent_metric_{i}", i * 10, datetime.now())
            self.metrics_store.save_model_metrics(metric)
        
        # Should handle all writes successfully
        metrics = self.metrics_store.get_model_metrics()
        concurrent_metrics = [m for m in metrics if "concurrent_metric" in m["name"]]
        self.assertEqual(len(concurrent_metrics), 10)
    
    def test_main_function(self):
        """Test main function execution."""
        from src.metrics_store_stub import main
        
        # Should not raise any exceptions
        try:
            main()
        except Exception as e:
            self.fail(f"Main function should not raise exceptions: {e}")


if __name__ == '__main__':
    unittest.main()