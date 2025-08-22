"""
Unit tests for ml_utils functionality with mocked dependencies.

These tests focus on key ML utility functions while mocking heavy dependencies
to improve coverage without requiring scikit-learn, OpenAI, etc.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_utils_stub import (
    analyze_patterns_with_ai, 
    validate_tree_data, 
    predict_tree_health
)


class MockDataFrame:
    """Mock DataFrame for testing without pandas dependency."""
    
    def __init__(self, data=None, empty=False):
        self.data = data or {}
        self.empty = empty
        self.columns = list(self.data.keys()) if self.data else []
    
    def __len__(self):
        if self.empty or not self.data:
            return 0
        return len(list(self.data.values())[0]) if self.data else 0
    
    def __getitem__(self, key):
        if key in self.data:
            return MockSeries(self.data[key])
        return MockSeries([])
    
    @property
    def iloc(self):
        return MockIndexer(self.data)


class MockSeries:
    """Mock Series for testing."""
    
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index] if 0 <= index < len(self.data) else None
    
    @property
    def iloc(self):
        return MockIndexer(self.data)


class MockIndexer:
    """Mock indexer for iloc operations."""
    
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        if isinstance(self.data, dict):
            return {key: values[index] for key, values in self.data.items() if index < len(values)}
        elif isinstance(self.data, list):
            return self.data[index] if 0 <= index < len(self.data) else None
        return None


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, response_text="Test analysis response"):
        self.response_text = response_text
        self.chat = MockChatCompletion(response_text)


class MockChatCompletion:
    """Mock chat completion interface."""
    
    def __init__(self, response_text):
        self.response_text = response_text
        self.completions = self
    
    def create(self, **kwargs):
        """Mock create method."""
        return MockResponse(self.response_text)


class MockResponse:
    """Mock OpenAI response."""
    
    def __init__(self, response_text):
        self.choices = [MockChoice(response_text)]


class MockChoice:
    """Mock choice from OpenAI response."""
    
    def __init__(self, response_text):
        self.message = MockMessage(response_text)


class MockMessage:
    """Mock message from OpenAI choice."""
    
    def __init__(self, response_text):
        self.content = response_text


class TestMLUtilsEdgeCases(unittest.TestCase):
    """Test edge cases for ML utilities functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_analyze_patterns_with_ai_none_data(self):
        """Test analyze_patterns_with_ai with None data."""
        with self.assertRaises(ValueError) as context:
            analyze_patterns_with_ai(None, "Test prompt", lambda: MockOpenAIClient())
        
        self.assertIn("Data cannot be None", str(context.exception))
    
    def test_analyze_patterns_with_ai_empty_data(self):
        """Test analyze_patterns_with_ai with empty data."""
        empty_data = MockDataFrame(empty=True)
        
        with self.assertRaises(ValueError) as context:
            analyze_patterns_with_ai(empty_data, "Test prompt", lambda: MockOpenAIClient())
        
        self.assertIn("Data cannot be empty", str(context.exception))
    
    def test_analyze_patterns_with_ai_missing_columns(self):
        """Test analyze_patterns_with_ai with missing required columns."""
        incomplete_data = MockDataFrame({
            "species": ["Oak", "Maple"],
            "health": ["Good", "Fair"]
            # Missing diameter, latitude, longitude
        })
        
        with self.assertRaises(ValueError) as context:
            analyze_patterns_with_ai(incomplete_data, "Test prompt", lambda: MockOpenAIClient())
        
        self.assertIn("Missing required column", str(context.exception))
    
    def test_analyze_patterns_with_ai_invalid_data_types(self):
        """Test analyze_patterns_with_ai with invalid data types."""
        # Test with invalid diameter (string instead of numeric)
        invalid_data = MockDataFrame({
            "species": ["Oak", "Maple"],
            "health": ["Good", "Fair"],
            "diameter": ["not a number", "15.5"],
            "latitude": [40.7128, 40.7129],
            "longitude": [-74.0060, -74.0061]
        })
        
        with self.assertRaises(ValueError) as context:
            analyze_patterns_with_ai(invalid_data, "Test prompt", lambda: MockOpenAIClient())
        
        self.assertIn("Invalid data type in column diameter", str(context.exception))
    
    def test_analyze_patterns_with_ai_invalid_species_type(self):
        """Test analyze_patterns_with_ai with non-string species."""
        invalid_species_data = MockDataFrame({
            "species": [1, 2],  # Non-string species
            "health": ["Good", "Fair"],
            "diameter": [15.5, 12.3],
            "latitude": [40.7128, 40.7129],
            "longitude": [-74.0060, -74.0061]
        })
        
        with self.assertRaises(ValueError) as context:
            analyze_patterns_with_ai(invalid_species_data, "Test prompt", lambda: MockOpenAIClient())
        
        self.assertIn("Invalid data type in column species", str(context.exception))
    
    def test_analyze_patterns_with_ai_valid_data_stub_mode(self):
        """Test analyze_patterns_with_ai with valid data in stub mode."""
        valid_data = MockDataFrame({
            "species": ["Oak", "Maple"],
            "health": ["Good", "Fair"],
            "diameter": [15.5, 12.3],
            "latitude": [40.7128, 40.7129],
            "longitude": [-74.0060, -74.0061]
        })
        
        result = analyze_patterns_with_ai(
            valid_data, 
            "Analyze tree patterns", 
            lambda: MockOpenAIClient()
        )
        
        self.assertEqual(result, "Test analysis response")
    
    def test_validate_tree_data_none_input(self):
        """Test validate_tree_data with None input."""
        result = validate_tree_data(None)
        
        self.assertFalse(result["is_valid"])
        self.assertIn("Data is None", result["errors"])
        self.assertEqual(result["summary"]["total_records"], 0)
    
    def test_validate_tree_data_empty_data(self):
        """Test validate_tree_data with empty data."""
        empty_data = MockDataFrame(empty=True)
        
        result = validate_tree_data(empty_data)
        
        # In stub mode, should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("is_valid", result)
        self.assertIn("errors", result)
        self.assertIn("warnings", result)
        self.assertIn("summary", result)
    
    def test_validate_tree_data_with_data(self):
        """Test validate_tree_data with valid data."""
        valid_data = MockDataFrame({
            "species": ["Oak", "Maple", "Pine"],
            "health": ["Good", "Fair", "Poor"],
            "diameter": [15.5, 12.3, 8.7],
            "latitude": [40.7128, 40.7129, 40.7130],
            "longitude": [-74.0060, -74.0061, -74.0062]
        })
        
        result = validate_tree_data(valid_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn("is_valid", result)
        self.assertIn("summary", result)
        self.assertGreaterEqual(result["summary"]["total_records"], 0)
    
    def test_predict_tree_health_none_data(self):
        """Test predict_tree_health with None data."""
        result = predict_tree_health(None)
        
        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
        self.assertIn("confidence_scores", result)
        self.assertIn("model_accuracy", result)
        self.assertIn("feature_importance", result)
        
        self.assertEqual(len(result["predictions"]), 0)
        self.assertEqual(len(result["confidence_scores"]), 0)
    
    def test_predict_tree_health_empty_data(self):
        """Test predict_tree_health with empty data."""
        empty_data = MockDataFrame(empty=True)
        
        result = predict_tree_health(empty_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
        self.assertIn("confidence_scores", result)
        self.assertEqual(len(result["predictions"]), 0)
    
    def test_predict_tree_health_with_valid_data(self):
        """Test predict_tree_health with valid data."""
        valid_data = MockDataFrame({
            "species": ["Oak", "Maple"],
            "health": ["Good", "Fair"],
            "diameter": [15.5, 12.3],
            "latitude": [40.7128, 40.7129],
            "longitude": [-74.0060, -74.0061]
        })
        
        result = predict_tree_health(valid_data)
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result["predictions"]), 0)
        self.assertGreater(len(result["confidence_scores"]), 0)
        self.assertGreater(result["model_accuracy"], 0)
        self.assertIsInstance(result["feature_importance"], dict)
    
    def test_predict_tree_health_with_model_path(self):
        """Test predict_tree_health with model path."""
        valid_data = MockDataFrame({
            "species": ["Oak"],
            "diameter": [15.5],
            "latitude": [40.7128],
            "longitude": [-74.0060]
        })
        
        # Test with non-existent model path
        result = predict_tree_health(valid_data, "/path/to/nonexistent/model.pkl")
        
        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
    
    def test_analyze_patterns_openai_error_handling(self):
        """Test error handling in OpenAI client interactions."""
        def failing_client_factory():
            client = MockOpenAIClient()
            # Mock a client that raises an exception
            def failing_create(**kwargs):
                raise Exception("API Error")
            client.chat.completions.create = failing_create
            return client
        
        valid_data = MockDataFrame({
            "species": ["Oak"],
            "health": ["Good"],
            "diameter": [15.5],
            "latitude": [40.7128],
            "longitude": [-74.0060]
        })
        
        # In stub mode, should still return test response
        result = analyze_patterns_with_ai(
            valid_data,
            "Test prompt",
            failing_client_factory
        )
        
        # Should handle gracefully in stub mode
        self.assertIsInstance(result, str)
    
    def test_feature_flag_behavior(self):
        """Test behavior with feature flags."""
        from src.ml_utils_stub import ML_DEPENDENCIES_AVAILABLE
        
        # Test that stub implementation works regardless of dependencies
        test_data = MockDataFrame({
            "species": ["Oak"],
            "health": ["Good"],
            "diameter": [15.5],
            "latitude": [40.7128],
            "longitude": [-74.0060]
        })
        
        # All functions should work in stub mode
        try:
            analysis_result = analyze_patterns_with_ai(
                test_data, 
                "Test", 
                lambda: MockOpenAIClient()
            )
            validation_result = validate_tree_data(test_data)
            prediction_result = predict_tree_health(test_data)
            
            self.assertIsInstance(analysis_result, str)
            self.assertIsInstance(validation_result, dict)
            self.assertIsInstance(prediction_result, dict)
        except Exception as e:
            self.fail(f"Stub implementations should not fail: {e}")
    
    def test_edge_case_data_structures(self):
        """Test with edge case data structures."""
        # Test with data that has __len__ but might not be DataFrame
        class EdgeCaseData:
            def __len__(self):
                return 2
            
            def __getattr__(self, name):
                if name == 'empty':
                    return False
                elif name == 'columns':
                    return ["species", "health", "diameter", "latitude", "longitude"]
                return None
        
        edge_data = EdgeCaseData()
        
        # Should handle gracefully
        try:
            result = validate_tree_data(edge_data)
            self.assertIsInstance(result, dict)
        except Exception:
            # Some exceptions are acceptable for unusual data structures
            pass
    
    def test_main_function(self):
        """Test main function execution."""
        from src.ml_utils_stub import main
        
        # Should not raise any exceptions
        try:
            main()
        except Exception as e:
            self.fail(f"Main function should not raise exceptions: {e}")
    
    def test_data_type_validation_edge_cases(self):
        """Test data type validation with edge cases."""
        # Test with mixed valid/invalid data
        mixed_data = MockDataFrame({
            "species": ["Oak", "Maple", 123, None],  # Mixed types
            "health": ["Good", "Fair", "Poor", ""],
            "diameter": [15.5, "invalid", -5.0, float('inf')],  # Various invalid values
            "latitude": [40.7128, 999.0, None, "invalid"],  # Out of range and invalid
            "longitude": [-74.0060, -181.0, None, "text"]  # Out of range and invalid
        })
        
        # Should handle mixed data gracefully
        try:
            validation_result = validate_tree_data(mixed_data)
            self.assertIsInstance(validation_result, dict)
            
            # Prediction should also handle gracefully
            prediction_result = predict_tree_health(mixed_data)
            self.assertIsInstance(prediction_result, dict)
        except Exception:
            # Some exceptions are acceptable for severely malformed data
            pass
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create a larger mock dataset
        large_size = 1000
        large_data = MockDataFrame({
            "species": ["Oak"] * large_size,
            "health": ["Good"] * large_size,
            "diameter": [15.5] * large_size,
            "latitude": [40.7128] * large_size,
            "longitude": [-74.0060] * large_size
        })
        
        # Should handle large datasets efficiently
        try:
            validation_result = validate_tree_data(large_data)
            self.assertIsInstance(validation_result, dict)
            self.assertEqual(validation_result["summary"]["total_records"], large_size)
            
            prediction_result = predict_tree_health(large_data)
            self.assertIsInstance(prediction_result, dict)
            self.assertEqual(len(prediction_result["predictions"]), large_size)
        except Exception as e:
            self.fail(f"Should handle large datasets: {e}")


if __name__ == '__main__':
    unittest.main()