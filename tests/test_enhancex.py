"""Tests for the EnhanceX memory management system."""

import os
import sys
import unittest
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory import EnhanceX, MemoryStore, MemoryType


class TestEnhanceX(unittest.TestCase):
	def setUp(self):
		self.test_dir = tempfile.mkdtemp()
		self.enhancex = EnhanceX(data_dir=self.test_dir)

	def tearDown(self):
		shutil.rmtree(self.test_dir)

	def test_session_management(self):
		session_id = self.enhancex.start_session({"test": True})
		self.assertTrue(session_id)
		self.assertTrue(self.enhancex.is_session_active())
		self.assertTrue(self.enhancex.end_session())
		self.assertFalse(self.enhancex.is_session_active())

	def test_user_preferences(self):
		self.enhancex.set_user_preference("ui", "theme", "dark")
		self.enhancex.set_user_preference("visualization", "color_palette", "viridis")
		theme = self.enhancex.get_user_preference("ui", "theme")
		palette = self.enhancex.get_user_preference("visualization", "color_palette")
		self.assertEqual(theme, "dark")
		self.assertEqual(palette, "viridis")

	def test_interaction_tracking(self):
		self.enhancex.start_session()
		self.enhancex.record_interaction("filter_change", {"filter": "city", "value": "New York"})
		self.enhancex.record_interaction("chart_click", {"chart": "species_distribution", "value": "Oak"})
		recent = self.enhancex.get_recent_interactions()
		self.assertEqual(len(recent), 2)
		self.enhancex.end_session()

	def test_session_state(self):
		self.enhancex.start_session()
		self.enhancex.update_session_state({"current_view": "map", "filters": {"city": "New York"}})
		state = self.enhancex.get_session_state()
		self.assertEqual(state["current_view"], "map")
		self.enhancex.end_session()

	def test_long_term_memory(self):
		self.enhancex.store_long_term_memory("test_data", {"value": "x", "timestamp": time.time()})
		data = self.enhancex.retrieve_long_term_memory("test_data")
		self.assertEqual(data["value"], "x")

	def test_interaction_handlers(self):
		self.enhancex.start_session()
		called = {"flag": False}

		def h(interaction):
			called["flag"] = True

		self.enhancex.register_interaction_handler("evt", h)
		self.enhancex.record_interaction("evt", {})
		self.assertTrue(called["flag"])
		self.enhancex.end_session()


if __name__ == "__main__":  # pragma: no cover
	unittest.main()