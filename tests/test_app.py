import unittest
from unittest.mock import patch, MagicMock
from app import app, server
from src.config import Config, ErrorConfig

class TestDashboardApp(unittest.TestCase):
    def setUp(self):
        self.app = app.server.test_client()
        self.app.testing = True

    def test_app_creation(self):
        """Test if the app is created successfully"""
        self.assertIsNotNone(app)
        self.assertIsNotNone(server)

    @patch('src.metrics.environmental_metrics.EnvironmentalMetrics')
    def test_environmental_metrics(self, mock_metrics):
        """Test environmental metrics visualization"""
        mock_metrics.return_value.get_visualization.return_value = {'data': []}
        response = self.app.get('/environmental-metrics')
        self.assertEqual(response.status_code, 200)

    @patch('src.metrics.model_metrics.ModelMetrics')
    def test_model_metrics(self, mock_metrics):
        """Test model metrics visualization"""
        mock_metrics.return_value.get_visualization.return_value = {'data': []}
        response = self.app.get('/model-metrics')
        self.assertEqual(response.status_code, 200)

    @patch('src.metrics.performance_metrics.PerformanceMetrics')
    def test_performance_metrics(self, mock_metrics):
        """Test performance metrics visualization"""
        mock_metrics.return_value.get_visualization.return_value = {'data': []}
        response = self.app.get('/performance-metrics')
        self.assertEqual(response.status_code, 200)

    def test_error_handling(self):
        """Test error handling functionality"""
        response = self.app.get('/nonexistent-route')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()