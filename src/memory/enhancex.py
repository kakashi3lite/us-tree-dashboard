"""EnhanceX Integration Module

Provides integration between the memory management system and the dashboard application.
This module serves as the main entry point for using EnhanceX features.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from .memory_store import MemoryStore, MemoryType
from .context_manager import ContextManager
from .preference_tracker import PreferenceTracker
from .session_memory import SessionMemory

logger = logging.getLogger(__name__)


class EnhanceX:
	"""Main integration class exposing memory + preference features."""

	def __init__(self, data_dir: Optional[str] = None, session_timeout: int = 30):
		# Determine persistent data directory
		if data_dir is None:
			project_root = Path(__file__).resolve().parents[2]
			data_dir = project_root / "data" / "memory"
		else:
			data_dir = Path(data_dir)
		data_dir.mkdir(parents=True, exist_ok=True)

		# Core components
		self.memory_store = MemoryStore(data_dir)
		self.context_manager = ContextManager(self.memory_store)
		self.preference_tracker = PreferenceTracker(self.memory_store)
		self.session_memory = SessionMemory(self.memory_store, session_timeout)

		if not self.preference_tracker.has_preferences():
			self._initialize_default_preferences()

		logger.info("EnhanceX initialized", extra={"data_dir": data_dir})

	# ------------------------------------------------------------------
	# Preference initialization
	# ------------------------------------------------------------------
	def _initialize_default_preferences(self) -> None:
		self.preference_tracker.create_category(
			"ui", "UI Preferences", "User interface display preferences"
		)
		self.preference_tracker.set_preference("ui", "theme", "light")
		self.preference_tracker.set_preference("ui", "sidebar_collapsed", False)
		self.preference_tracker.set_preference("ui", "map_style", "streets")

		self.preference_tracker.create_category(
			"visualization", "Visualization Preferences", "Chart / graph display preferences"
		)
		self.preference_tracker.set_preference("visualization", "default_chart_type", "bar")
		self.preference_tracker.set_preference("visualization", "color_palette", "viridis")
		self.preference_tracker.set_preference("visualization", "show_grid_lines", True)

		self.preference_tracker.create_category(
			"notifications", "Notification Preferences", "User notification settings"
		)
		self.preference_tracker.set_preference("notifications", "show_alerts", True)
		self.preference_tracker.set_preference(
			"notifications", "notification_position", "top-right"
		)

	# ------------------------------------------------------------------
	# Session operations passthrough
	# ------------------------------------------------------------------
	def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
		return self.session_memory.start_session(metadata)

	def end_session(self) -> bool:
		return self.session_memory.end_session()

	def is_session_active(self) -> bool:
		return self.session_memory.is_session_active()

	def keep_session_alive(self) -> bool:
		return self.session_memory.keep_alive()

	def get_session_duration(self) -> Optional[float]:
		return self.session_memory.get_session_duration()

	def record_interaction(
		self, interaction_type: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
	) -> Optional[str]:
		return self.session_memory.record_interaction(interaction_type, data, metadata)

	def get_recent_interactions(
		self, interaction_type: Optional[str] = None, limit: int = 10
	) -> List[Dict[str, Any]]:
		return self.session_memory.get_recent_interactions(interaction_type, limit)

	def register_interaction_handler(self, interaction_type: str, handler: Callable) -> None:
		self.session_memory.register_interaction_handler(interaction_type, handler)

	def get_session_state(self, key: Optional[str] = None) -> Any:
		return self.session_memory.get_state(key)

	def update_session_state(self, updates: Dict[str, Any]) -> bool:
		return self.session_memory.update_state(updates)

	# ------------------------------------------------------------------
	# Preferences
	# ------------------------------------------------------------------
	def get_user_preference(self, category: str, key: str, default: Any = None) -> Any:
		value = self.preference_tracker.get_preference(category, key)
		return default if value is None else value

	def set_user_preference(self, category: str, key: str, value: Any) -> bool:
		return self.preference_tracker.set_preference(category, key, value)

	def get_all_preferences(self) -> Dict[str, Dict[str, Any]]:
		return self.preference_tracker.get_all_preferences()

	# ------------------------------------------------------------------
	# Project Context
	# ------------------------------------------------------------------
	def create_project_context(
		self, name: str, description: str = "", data: Optional[Dict[str, Any]] = None
	) -> str:
		return self.context_manager.create_context(name, description, data)

	def get_project_context(self, context_id: str) -> Optional[Dict[str, Any]]:
		ctx = self.context_manager.get_context(context_id)
		return ctx.data if ctx else None

	def update_project_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
		return self.context_manager.update_context(context_id, updates)

	# ------------------------------------------------------------------
	# Long-term memory passthrough
	# ------------------------------------------------------------------
	def store_long_term_memory(
		self, key: str, value: Any, expiry: Optional[float] = None
	) -> bool:
		return self.memory_store.store(key, value, MemoryType.LONG_TERM, expiry)

	def retrieve_long_term_memory(self, key: str) -> Any:
		return self.memory_store.retrieve(key, MemoryType.LONG_TERM)

	def cleanup_expired_memory(self) -> int:
		return self.memory_store.cleanup_expired()