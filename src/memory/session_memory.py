"""Session memory management for EnhanceX (clean version)."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import time
import logging
import uuid

from .memory_store import MemoryStore, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
	interaction_id: str
	interaction_type: str
	timestamp: float
	data: Dict[str, Any]
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
	session_id: str
	start_time: float
	last_activity: float
	state: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)


class SessionMemory:
	def __init__(self, memory_store: MemoryStore, session_timeout: int = 30):
		self.memory_store = memory_store
		self.session_timeout = session_timeout
		self.current_session: Optional[SessionState] = None
		self.interactions: List[Interaction] = []
		self.interaction_handlers: Dict[str, List[Callable]] = {}
		self._restore_session()

	def _restore_session(self) -> bool:
		data = self.memory_store.retrieve("current_session", MemoryType.SHORT_TERM)
		if not data:
			return False
		last_activity = data.get("last_activity", 0)
		if time.time() - last_activity > (self.session_timeout * 60):
			return False
		self.current_session = SessionState(
			session_id=data["session_id"],
			start_time=data["start_time"],
			last_activity=data["last_activity"],
			state=data["state"],
			metadata=data["metadata"],
		)
		interactions_data = self.memory_store.retrieve(
			f"interactions_{self.current_session.session_id}", MemoryType.SHORT_TERM
		)
		if interactions_data:
			self.interactions = [
				Interaction(
					interaction_id=i["interaction_id"],
					interaction_type=i["interaction_type"],
					timestamp=i["timestamp"],
					data=i["data"],
					metadata=i["metadata"],
				)
				for i in interactions_data
			]
		return True

	def _save_session(self):
		if not self.current_session:
			return
		self.current_session.last_activity = time.time()
		self.memory_store.store(
			"current_session",
			{
				"session_id": self.current_session.session_id,
				"start_time": self.current_session.start_time,
				"last_activity": self.current_session.last_activity,
				"state": self.current_session.state,
				"metadata": self.current_session.metadata,
			},
			MemoryType.SHORT_TERM,
			expiry=time.time() + ((self.session_timeout + 5) * 60),
		)
		if self.interactions:
			self.memory_store.store(
				f"interactions_{self.current_session.session_id}",
				[
					{
						"interaction_id": i.interaction_id,
						"interaction_type": i.interaction_type,
						"timestamp": i.timestamp,
						"data": i.data,
						"metadata": i.metadata,
					}
					for i in self.interactions
				],
				MemoryType.SHORT_TERM,
				expiry=time.time() + ((self.session_timeout + 5) * 60),
			)

	def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
		sid = str(uuid.uuid4())
		now = time.time()
		self.current_session = SessionState(
			session_id=sid,
			start_time=now,
			last_activity=now,
			metadata=metadata or {},
		)
		self.interactions = []
		self._save_session()
		return sid

	def end_session(self) -> bool:
		if not self.current_session:
			return False
		self.memory_store.delete("current_session", MemoryType.SHORT_TERM)
		self.memory_store.delete(
			f"interactions_{self.current_session.session_id}", MemoryType.SHORT_TERM
		)
		self.current_session = None
		self.interactions = []
		return True

	def record_interaction(
		self, interaction_type: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
	) -> Optional[str]:
		if not self.current_session:
			logger.warning("Cannot record interaction: No active session")
			return None
		iid = str(uuid.uuid4())
		self.interactions.append(
			Interaction(
				interaction_id=iid,
				interaction_type=interaction_type,
				timestamp=time.time(),
				data=data,
				metadata=metadata or {},
			)
		)
		if len(self.interactions) > 1000:
			self.interactions = self.interactions[-1000:]
		self._save_session()
		self._trigger_handlers(self.interactions[-1])
		return iid

	def get_state(self, key: Optional[str] = None):
		if not self.current_session:
			return None
		return self.current_session.state.get(key) if key else self.current_session.state.copy()

	def update_state(self, updates: Dict[str, Any]) -> bool:
		if not self.current_session:
			return False
		self.current_session.state.update(updates)
		self._save_session()
		return True

	def get_recent_interactions(self, interaction_type: Optional[str] = None, limit: int = 10):
		items = self.interactions
		if interaction_type:
			items = [i for i in items if i.interaction_type == interaction_type]
		return [
			{
				"interaction_id": i.interaction_id,
				"interaction_type": i.interaction_type,
				"timestamp": i.timestamp,
				"data": i.data,
				"metadata": i.metadata,
			}
			for i in sorted(items, key=lambda x: x.timestamp, reverse=True)[:limit]
		]

	def register_interaction_handler(self, interaction_type: str, handler: Callable):
		self.interaction_handlers.setdefault(interaction_type, []).append(handler)

	def _trigger_handlers(self, interaction: Interaction):
		for h in self.interaction_handlers.get(interaction.interaction_type, []):
			try:
				h(interaction)
			except Exception as e:
				logger.error("Error in interaction handler: %s", e)

	def get_session_duration(self) -> Optional[float]:
		if not self.current_session:
			return None
		return time.time() - self.current_session.start_time

	def is_session_active(self) -> bool:
		if not self.current_session:
			return False
		if time.time() - self.current_session.last_activity > (self.session_timeout * 60):
			self.end_session()
			return False
		return True

	def keep_alive(self) -> bool:
		if not self.current_session:
			return False
		self.current_session.last_activity = time.time()
		self._save_session()
		return True
