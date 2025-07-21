"""
Context management for EnhanceX.

Provides functionality to track and manage project context,
including user interactions, project state, and task history.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import logging

from .memory_store import MemoryStore, MemoryType

logger = logging.getLogger(__name__)

@dataclass
class ContextItem:
    """A single context item with metadata."""
    context_type: str  # e.g., 'filter_selection', 'view_state', 'analysis_result'
    value: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectContext:
    """Project context container."""
    project_id: str
    name: str
    items: Dict[str, ContextItem] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "items": {k: {
                "context_type": v.context_type,
                "value": v.value,
                "timestamp": v.timestamp,
                "metadata": v.metadata
            } for k, v in self.items.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectContext':
        """Create project context from dictionary."""
        items = {}
        for k, v in data.get("items", {}).items():
            items[k] = ContextItem(
                context_type=v["context_type"],
                value=v["value"],
                timestamp=v["timestamp"],
                metadata=v["metadata"]
            )
        
        return cls(
            project_id=data["project_id"],
            name=data["name"],
            items=items,
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        )

class ContextManager:
    """Manages project context and state."""
    
    def __init__(self, memory_store: MemoryStore):
        """Initialize context manager with memory store."""
        self.memory_store = memory_store
        self.active_context: Optional[ProjectContext] = None
        self.context_history: List[str] = []
        
        # Load active context if available
        active_context_id = self.memory_store.retrieve("active_context_id", MemoryType.CONTEXT)
        if active_context_id:
            self.load_context(active_context_id)
    
    def create_context(self, name: str, project_id: Optional[str] = None) -> ProjectContext:
        """Create a new project context."""
        import uuid
        
        project_id = project_id or str(uuid.uuid4())
        context = ProjectContext(project_id=project_id, name=name)
        
        # Store in memory
        self.memory_store.store(
            key=project_id,
            value=context.to_dict(),
            memory_type=MemoryType.CONTEXT
        )
        
        # Set as active context
        self.active_context = context
        self.memory_store.store(
            key="active_context_id",
            value=project_id,
            memory_type=MemoryType.CONTEXT
        )
        
        # Update history
        self._update_context_history(project_id)
        
        return context
    
    def load_context(self, project_id: str) -> Optional[ProjectContext]:
        """Load a project context by ID."""
        context_data = self.memory_store.retrieve(project_id, MemoryType.CONTEXT)
        if not context_data:
            return None
        
        context = ProjectContext.from_dict(context_data)
        self.active_context = context
        
        # Update active context reference
        self.memory_store.store(
            key="active_context_id",
            value=project_id,
            memory_type=MemoryType.CONTEXT
        )
        
        # Update history
        self._update_context_history(project_id)
        
        return context
    
    def _update_context_history(self, project_id: str):
        """Update context history."""
        # Remove if already in history
        if project_id in self.context_history:
            self.context_history.remove(project_id)
        
        # Add to front of history
        self.context_history.insert(0, project_id)
        
        # Limit history length
        self.context_history = self.context_history[:10]
        
        # Store updated history
        self.memory_store.store(
            key="context_history",
            value=self.context_history,
            memory_type=MemoryType.CONTEXT
        )
    
    def get_context_history(self) -> List[Dict[str, Any]]:
        """Get list of recent contexts with basic info."""
        history = []
        
        for project_id in self.context_history:
            context_data = self.memory_store.retrieve(project_id, MemoryType.CONTEXT)
            if context_data:
                history.append({
                    "project_id": context_data["project_id"],
                    "name": context_data["name"],
                    "updated_at": context_data["updated_at"]
                })
        
        return history
    
    def update_context(self, key: str, context_type: str, value: Any, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a context item in the active context."""
        if not self.active_context:
            logger.warning("No active context to update")
            return False
        
        # Update or create context item
        self.active_context.items[key] = ContextItem(
            context_type=context_type,
            value=value,
            metadata=metadata or {}
        )
        
        # Update timestamp
        self.active_context.updated_at = time.time()
        
        # Store updated context
        self.memory_store.store(
            key=self.active_context.project_id,
            value=self.active_context.to_dict(),
            memory_type=MemoryType.CONTEXT
        )
        
        return True
    
    def get_context_item(self, key: str) -> Optional[Any]:
        """Get a context item value from the active context."""
        if not self.active_context or key not in self.active_context.items:
            return None
        
        return self.active_context.items[key].value
    
    def get_context_items_by_type(self, context_type: str) -> Dict[str, Any]:
        """Get all context items of a specific type."""
        if not self.active_context:
            return {}
        
        return {k: v.value for k, v in self.active_context.items.items() 
                if v.context_type == context_type}
    
    def delete_context(self, project_id: str) -> bool:
        """Delete a project context."""
        # Remove from memory store
        result = self.memory_store.delete(project_id, MemoryType.CONTEXT)
        
        # Update history if needed
        if project_id in self.context_history:
            self.context_history.remove(project_id)
            self.memory_store.store(
                key="context_history",
                value=self.context_history,
                memory_type=MemoryType.CONTEXT
            )
        
        # Clear active context if it was deleted
        if self.active_context and self.active_context.project_id == project_id:
            self.active_context = None
            self.memory_store.delete("active_context_id", MemoryType.CONTEXT)
        
        return result