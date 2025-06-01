"""Integration tests for notebook analysis functions"""
import unittest
import pandas as pd
import os
from src.ml_utils import analyze_patterns_with_ai
from unittest.mock import MagicMock

class MockOpenAI:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=type('MockResponse', (), {
            'choices': [type('MockChoice', (), {'message': type('MockMessage', (), {'content': 'Test analysis'})()})]
        }))

class TestNotebookAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create test data directory
        os.makedirs('data/test', exist_ok=True)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'species': ['Oak', 'Maple', 'Oak', 'Pine'] * 25,  # 100 rows
            'health': ['Good', 'Fair', 'Good', 'Poor'] * 25,
            'diameter': [10.5, 15.2, 12.1, 8.7] * 25,
            'latitude': [40.7128, 40.7129, 40.7130, 40.7131] * 25,
            'longitude': [-74.0060, -74.0061, -74.0062, -74.0063] * 25
        })
        
        # Save test data
        self.test_data.to_csv('data/test/test_trees.csv', index=False)

    def test_tree_analysis(self):
        """Test tree analysis functionality"""
        # Test analyzing species distribution
        result = analyze_patterns_with_ai(
            self.test_data, 
            "Analyze species distribution",
            lambda: MockOpenAI()
        )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'Test analysis')

    def tearDown(self):
        """Clean up test data"""
        if os.path.exists('data/test/test_trees.csv'):
            os.remove('data/test/test_trees.csv')
        if os.path.exists('data/test'):
            os.rmdir('data/test')

if __name__ == '__main__':
    unittest.main()
