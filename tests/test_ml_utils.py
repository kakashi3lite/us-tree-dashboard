import unittest
import pandas as pd
import numpy as np
from src.ml_utils import analyze_patterns_with_ai
from unittest.mock import MagicMock
from dataclasses import dataclass

@dataclass
class MockMessage:
    content: str

@dataclass
class MockChoice:
    message: MockMessage

@dataclass
class MockResponse:
    choices: list

class MockOpenAI:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=MockResponse([
            MockChoice(MockMessage("Test analysis response"))
        ]))

class TestMLUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'species': ['Oak', 'Maple', 'Oak', 'Pine'],
            'health': ['Good', 'Fair', 'Good', 'Poor'],
            'diameter': [10.5, 15.2, 12.1, 8.7],
            'latitude': [40.7128, 40.7129, 40.7130, 40.7131],
            'longitude': [-74.0060, -74.0061, -74.0062, -74.0063]
        })

    def test_analyze_patterns(self):
        """Test AI pattern analysis"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            analyze_patterns_with_ai(empty_df, "Analyze species distribution", lambda: MockOpenAI())

        # Test with valid data
        result = analyze_patterns_with_ai(self.test_data, "Analyze species distribution", lambda: MockOpenAI())
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result, "Test analysis response")

    def test_data_validation(self):
        """Test input data validation"""
        # Test with missing required columns
        bad_data = pd.DataFrame({'species': ['Oak'], 'health': ['Good']})
        with self.assertRaises(ValueError) as ctx:
            analyze_patterns_with_ai(bad_data, "Analyze patterns", lambda: MockOpenAI())
        self.assertTrue(any('Missing required column' in str(err) for err in [ctx.exception]))

        # Test with invalid data types
        invalid_data = self.test_data.copy()
        invalid_data['diameter'] = 'not a number'
        with self.assertRaises(ValueError) as ctx:
            analyze_patterns_with_ai(invalid_data, "Analyze patterns", lambda: MockOpenAI())
        self.assertTrue('Invalid data type in column diameter' in str(ctx.exception))

        # Test with non-string species
        invalid_data = self.test_data.copy()
        invalid_data['species'] = [1, 2, 3, 4]
        with self.assertRaises(ValueError) as ctx:
            analyze_patterns_with_ai(invalid_data, "Analyze patterns", lambda: MockOpenAI())
        self.assertTrue('Invalid data type in column species' in str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
