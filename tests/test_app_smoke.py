"""Smoke tests for core Dash app components and data transformations."""
import os
import importlib
import unittest
import pandas as pd

from pathlib import Path


class TestAppSmoke(unittest.TestCase):
    def setUp(self):
        # Ensure env vars minimal
        os.environ.setdefault("MAPBOX_TOKEN", "test-token")

    def test_app_imports(self):
        """Import the Dash app and ensure server + layout created."""
        app_mod = importlib.import_module("app")
        self.assertTrue(hasattr(app_mod, "app"))
        self.assertTrue(hasattr(app_mod, "server"))
        # Layout should not be None
        self.assertIsNotNone(app_mod.app.layout)

    def test_tree_dashboard_sample(self):
        """Instantiate TreeDashboard and verify sample fallback works."""
        from app import TreeDashboard
        dash_obj = TreeDashboard()
        # DataFrames created
        self.assertIsNotNone(dash_obj.df_trees)
        self.assertFalse(dash_obj.df_trees.empty)
        # Dropdown options prepared
        self.assertGreater(len(dash_obj.state_options), 0)
        self.assertGreater(len(dash_obj.city_options), 0)

    def test_map_creation_default(self):
        from app import TreeDashboard
        dash_obj = TreeDashboard()
        fig_component = dash_obj.create_map(None, None, [0, 100])
        # Graph object expected
        self.assertTrue(hasattr(fig_component, 'figure'))

    def test_summary_cards_empty(self):
        from app import TreeDashboard
        dash_obj = TreeDashboard()
        cards = dash_obj.create_summary_cards(None, None)
        self.assertIsInstance(cards, list)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
