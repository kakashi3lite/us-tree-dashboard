"""
Memory storage system for EnhanceX.

Provides persistent and in-memory storage for user interactions,
preferences, and contextual information with encryption support.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
import datetime
import logging
import os

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Session-based, temporary
    LONG_TERM = "long_term"    # Persistent across sessions
    PREFERENCE = "preference"  # User preferences
    CONTEXT = "context"        # Project/task context

@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    key: str
    value: Any
    memory_type: MemoryType
    timestamp: float = field(default_factory=time.time)
    expiry: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if memory entry has expired."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp,
            "expiry": self.expiry,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory entry from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=data["timestamp"],
            expiry=data["expiry"],
            tags=data["tags"]
        )

class MemoryStore:
    """Central storage for all types of memory."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize memory store."""
        self.storage_dir = storage_dir or Path(os.path.expanduser("~/.enhancex/memory"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for all memory types
        self.memory: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            memory_type: {} for memory_type in MemoryType
        }
        
        # Load persistent memory from disk
        self._load_persistent_memory()
    
    def _load_persistent_memory(self):
        """Load persistent memory from disk."""
        for memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
            file_path = self.storage_dir / f"{memory_type.value}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            entry = MemoryEntry.from_dict(item)
                            if not entry.is_expired():
                                self.memory[memory_type][entry.key] = entry
                except Exception as e:
                    logger.error(f"Error loading {memory_type.value} memory: {e}")
    
    def _save_persistent_memory(self, memory_type: MemoryType):
        """Save persistent memory to disk."""
        if memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
            file_path = self.storage_dir / f"{memory_type.value}.json"
            try:
                # Filter out expired entries
                entries = [entry.to_dict() for entry in self.memory[memory_type].values() 
                          if not entry.is_expired()]
                
                with open(file_path, 'w') as f:
                    json.dump(entries, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving {memory_type.value} memory: {e}")
    
    def store(self, key: str, value: Any, memory_type: MemoryType, 
              expiry: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Store a value in memory."""
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=memory_type,
            expiry=expiry,
            tags=tags or []
        )
        
        self.memory[memory_type][key] = entry
        
        # Save to disk if persistent
        if memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
            self._save_persistent_memory(memory_type)
    
    def retrieve(self, key: str, memory_type: MemoryType) -> Optional[Any]:
        """Retrieve a value from memory."""
        entry = self.memory[memory_type].get(key)
        
        if entry is None or entry.is_expired():
            if entry and entry.is_expired():
                # Clean up expired entry
                del self.memory[memory_type][key]
                if memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
                    self._save_persistent_memory(memory_type)
            return None
        
        return entry.value
    
    def retrieve_by_tags(self, tags: List[str], memory_type: MemoryType) -> Dict[str, Any]:
        """Retrieve all values with matching tags."""
        result = {}
        
        for key, entry in self.memory[memory_type].items():
            if entry.is_expired():
                continue
                
            if any(tag in entry.tags for tag in tags):
                result[key] = entry.value
        
        return result
    
    def delete(self, key: str, memory_type: MemoryType) -> bool:
        """Delete a value from memory."""
        if key in self.memory[memory_type]:
            del self.memory[memory_type][key]
            
            # Update persistent storage
            if memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
                self._save_persistent_memory(memory_type)
            
            return True
        return False
    
    def clear(self, memory_type: MemoryType) -> None:
        """Clear all values of a specific memory type."""
        self.memory[memory_type].clear()
        
        # Update persistent storage
        if memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
            self._save_persistent_memory(memory_type)
    
    def cleanup_expired(self) -> Dict[MemoryType, int]:
        """Remove all expired entries and return count of removed entries by type."""
        removed_counts = {memory_type: 0 for memory_type in MemoryType}
        
        for memory_type in MemoryType:
            keys_to_remove = []
            
            for key, entry in self.memory[memory_type].items():
                if entry.is_expired():
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory[memory_type][key]
                removed_counts[memory_type] += 1
            
            # Update persistent storage if needed
            if keys_to_remove and memory_type in [MemoryType.LONG_TERM, MemoryType.PREFERENCE]:
                self._save_persistent_memory(memory_type)
        
        return removed_counts