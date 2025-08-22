"""Preference tracking for EnhanceX.

Simplified clean version without escaped newlines.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time
import logging

from .memory_store import MemoryStore, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class PreferenceCategory:
	name: str
	description: str
	preferences: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	last_updated: float = field(default_factory=time.time)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"name": self.name,
			"description": self.description,
			"preferences": self.preferences,
			"metadata": self.metadata,
			"last_updated": self.last_updated,
		}

	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "PreferenceCategory":
		return cls(
			name=data["name"],
			description=data["description"],
			preferences=data.get("preferences", {}),
			metadata=data.get("metadata", {}),
			last_updated=data.get("last_updated", time.time()),
		)


class PreferenceTracker:
	def __init__(self, memory_store: MemoryStore):
		self.memory_store = memory_store
		self.categories: Dict[str, PreferenceCategory] = {}
		self._initialize_default_categories()
		self._load_preferences()

	def _initialize_default_categories(self):
		defaults = [
			PreferenceCategory("ui", "User interface", {"theme": "light", "layout": "default"}),
			PreferenceCategory("filters", "Filtering", {"canopy_range": [0, 100]}),
			PreferenceCategory("visualization", "Visualization settings", {"color_palette": "viridis"}),
		]
		for cat in defaults:
			self.categories[cat.name] = cat

	def _load_preferences(self):
		for name in list(self.categories.keys()):
			saved = self.memory_store.retrieve(f"preference_category_{name}", MemoryType.PREFERENCE)
			if saved:
				self.categories[name] = PreferenceCategory.from_dict(saved)

	def _save_category(self, name: str):
		if name in self.categories:
			cat = self.categories[name]
			cat.last_updated = time.time()
			self.memory_store.store(
				f"preference_category_{name}", cat.to_dict(), MemoryType.PREFERENCE
			)

	def get_preference(self, category: str, key: str):
		cat = self.categories.get(category)
		return None if not cat else cat.preferences.get(key)

	def set_preference(self, category: str, key: str, value):
		cat = self.categories.get(category)
		if not cat:
			logger.warning("Preference category '%s' does not exist", category)
			return False
		cat.preferences[key] = value
		self._save_category(category)
		return True

	def get_all_preferences(self) -> Dict[str, Dict[str, Any]]:
		return {k: v.preferences.copy() for k, v in self.categories.items()}

	def has_preferences(self) -> bool:
		"""Return True if any category has at least one stored preference."""
		return any(bool(cat.preferences) for cat in self.categories.values())

	def apply_preferences_to_context(self, context_key: str, preferences: Dict[str, Any]) -> None:
		logger.info("Applied preferences to context '%s': %s", context_key, preferences)