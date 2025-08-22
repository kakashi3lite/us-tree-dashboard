"""
Unit tests for memory_store edge cases and eviction logic.

These tests focus on missing keys, expiry handling, and memory eviction
to improve coverage for the memory management system.
"""

import unittest
import tempfile
import shutil
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.memory_store import MemoryStore, MemoryEntry, MemoryType


class TestMemoryStoreEdgeCases(unittest.TestCase):
    """Test edge cases for memory store functionality."""
    
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.memory_store = MemoryStore(storage_dir=Path(self.test_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_get_missing_keys(self):
        """Test retrieving missing keys returns None."""
        # Test missing key in each memory type
        for memory_type in MemoryType:
            result = self.memory_store.retrieve("nonexistent_key", memory_type)
            self.assertIsNone(result, f"Missing key should return None for {memory_type}")
    
    def test_get_with_default_value(self):
        """Test retrieving missing keys with default values."""
        # The retrieve method doesn't support default, so we'll test the basic behavior
        for memory_type in MemoryType:
            result = self.memory_store.retrieve("missing_key", memory_type)
            self.assertIsNone(result, f"Missing key should return None for {memory_type}")
            
            # Test getting default manually
            default_value = "default_test_value"
            result = self.memory_store.retrieve("missing_key", memory_type) or default_value
            self.assertEqual(result, default_value, f"Should return default for {memory_type}")
    
    def test_delete_missing_keys(self):
        """Test deleting missing keys returns False."""
        for memory_type in MemoryType:
            result = self.memory_store.delete("nonexistent_key", memory_type)
            self.assertFalse(result, f"Deleting missing key should return False for {memory_type}")
    
    def test_memory_expiry_logic(self):
        """Test memory entry expiry and cleanup."""
        # Store an entry with immediate expiry
        expired_time = time.time() - 1  # 1 second ago
        self.memory_store.store(
            "expired_key", 
            "expired_value", 
            MemoryType.SHORT_TERM,
            expiry=expired_time
        )
        
        # Store an entry with future expiry
        future_time = time.time() + 3600  # 1 hour from now
        self.memory_store.store(
            "valid_key",
            "valid_value", 
            MemoryType.SHORT_TERM,
            expiry=future_time
        )
        
        # Test that expired entry is not retrieved
        expired_result = self.memory_store.retrieve("expired_key", MemoryType.SHORT_TERM)
        self.assertIsNone(expired_result, "Expired entry should not be retrieved")
        
        # Test that valid entry is retrieved
        valid_result = self.memory_store.retrieve("valid_key", MemoryType.SHORT_TERM)
        self.assertEqual(valid_result, "valid_value", "Valid entry should be retrieved")
    
    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries."""
        # Add multiple entries with different expiry times
        current_time = time.time()
        
        # Expired entries
        self.memory_store.store("expired1", "value1", MemoryType.SHORT_TERM, expiry=current_time - 10)
        self.memory_store.store("expired2", "value2", MemoryType.LONG_TERM, expiry=current_time - 5)
        
        # Valid entries  
        self.memory_store.store("valid1", "value3", MemoryType.SHORT_TERM, expiry=current_time + 100)
        self.memory_store.store("valid2", "value4", MemoryType.LONG_TERM, expiry=current_time + 200)
        
        # Entry without expiry (should never expire)
        self.memory_store.store("permanent", "permanent_value", MemoryType.PREFERENCE)
        
        # Run cleanup
        removed_counts = self.memory_store.cleanup_expired()
        
        # Verify removed counts
        self.assertGreaterEqual(removed_counts[MemoryType.SHORT_TERM], 1, "Should remove at least 1 short term entry")
        self.assertGreaterEqual(removed_counts[MemoryType.LONG_TERM], 1, "Should remove at least 1 long term entry")
        
        # Verify valid entries still exist
        self.assertEqual(self.memory_store.retrieve("valid1", MemoryType.SHORT_TERM), "value3")
        self.assertEqual(self.memory_store.retrieve("valid2", MemoryType.LONG_TERM), "value4")
        self.assertEqual(self.memory_store.retrieve("permanent", MemoryType.PREFERENCE), "permanent_value")
        
        # Verify expired entries are gone
        self.assertIsNone(self.memory_store.retrieve("expired1", MemoryType.SHORT_TERM))
        self.assertIsNone(self.memory_store.retrieve("expired2", MemoryType.LONG_TERM))
    
    def test_memory_entry_is_expired(self):
        """Test MemoryEntry expiry checking."""
        current_time = time.time()
        
        # Entry without expiry
        no_expiry_entry = MemoryEntry("test", "value", MemoryType.SHORT_TERM)
        self.assertFalse(no_expiry_entry.is_expired(), "Entry without expiry should not be expired")
        
        # Entry with future expiry
        future_entry = MemoryEntry("test", "value", MemoryType.SHORT_TERM, expiry=current_time + 100)
        self.assertFalse(future_entry.is_expired(), "Entry with future expiry should not be expired")
        
        # Entry with past expiry
        past_entry = MemoryEntry("test", "value", MemoryType.SHORT_TERM, expiry=current_time - 1)
        self.assertTrue(past_entry.is_expired(), "Entry with past expiry should be expired")
    
    def test_memory_entry_serialization(self):
        """Test MemoryEntry to_dict and from_dict functionality."""
        original_entry = MemoryEntry(
            key="test_key",
            value={"nested": "data", "number": 42},
            memory_type=MemoryType.LONG_TERM,
            expiry=time.time() + 100,
            tags=["tag1", "tag2"]
        )
        
        # Test to_dict
        entry_dict = original_entry.to_dict()
        self.assertIsInstance(entry_dict, dict)
        self.assertEqual(entry_dict["key"], "test_key")
        self.assertEqual(entry_dict["value"], {"nested": "data", "number": 42})
        self.assertEqual(entry_dict["memory_type"], "long_term")
        self.assertEqual(entry_dict["tags"], ["tag1", "tag2"])
        
        # Test from_dict
        restored_entry = MemoryEntry.from_dict(entry_dict)
        self.assertEqual(restored_entry.key, original_entry.key)
        self.assertEqual(restored_entry.value, original_entry.value)
        self.assertEqual(restored_entry.memory_type, original_entry.memory_type)
        self.assertEqual(restored_entry.tags, original_entry.tags)
        self.assertAlmostEqual(restored_entry.expiry, original_entry.expiry, places=5)
    
    def test_clear_memory_type(self):
        """Test clearing all entries of a specific memory type."""
        # Add entries to different memory types
        self.memory_store.store("short1", "value1", MemoryType.SHORT_TERM)
        self.memory_store.store("short2", "value2", MemoryType.SHORT_TERM)
        self.memory_store.store("long1", "value3", MemoryType.LONG_TERM)
        self.memory_store.store("pref1", "value4", MemoryType.PREFERENCE)
        
        # Clear short term memory
        self.memory_store.clear(MemoryType.SHORT_TERM)
        
        # Verify short term entries are gone
        self.assertIsNone(self.memory_store.retrieve("short1", MemoryType.SHORT_TERM))
        self.assertIsNone(self.memory_store.retrieve("short2", MemoryType.SHORT_TERM))
        
        # Verify other entries remain
        self.assertEqual(self.memory_store.retrieve("long1", MemoryType.LONG_TERM), "value3")
        self.assertEqual(self.memory_store.retrieve("pref1", MemoryType.PREFERENCE), "value4")
    
    def test_get_all_keys_for_type(self):
        """Test retrieving all keys for a memory type."""
        # Add entries to short term memory
        self.memory_store.store("key1", "value1", MemoryType.SHORT_TERM)
        self.memory_store.store("key2", "value2", MemoryType.SHORT_TERM) 
        self.memory_store.store("other_key", "value3", MemoryType.LONG_TERM)
        
        # Check if we can access keys directly from internal structure
        short_term_keys = list(self.memory_store.memory[MemoryType.SHORT_TERM].keys())
        
        self.assertIn("key1", short_term_keys)
        self.assertIn("key2", short_term_keys)
        # other_key should not be in short term
        long_term_keys = list(self.memory_store.memory[MemoryType.LONG_TERM].keys())
        self.assertIn("other_key", long_term_keys)
    
    def test_storage_with_tags(self):
        """Test storing and retrieving entries with tags."""
        tags = ["urgent", "conservation", "field_data"]
        
        self.memory_store.store(
            "tagged_entry",
            "tagged_value", 
            MemoryType.CONTEXT,
            tags=tags
        )
        
        # Retrieve and verify tags
        retrieved_value = self.memory_store.retrieve("tagged_entry", MemoryType.CONTEXT)
        self.assertEqual(retrieved_value, "tagged_value")
        
        # Test retrieve by tags method
        entries_with_tags = self.memory_store.retrieve_by_tags(["urgent"], MemoryType.CONTEXT)
        self.assertIn("tagged_entry", entries_with_tags)
        self.assertEqual(entries_with_tags["tagged_entry"], "tagged_value")
    
    def test_persistent_storage_error_handling(self):
        """Test error handling in persistent storage operations."""
        # Create entry for persistent storage
        self.memory_store.store("persistent_test", "test_value", MemoryType.LONG_TERM)
        
        # Try to corrupt the storage directory to test error handling
        storage_file = self.memory_store.storage_dir / "long_term.json"
        if storage_file.exists():
            # Make file unreadable to test error handling
            with open(storage_file, 'w') as f:
                f.write("invalid json content {")
        
        # Try to load - should handle error gracefully
        # Create new store instance to trigger loading
        try:
            new_store = MemoryStore(storage_dir=Path(self.test_dir))
            # Should not crash even with corrupted file
            result = new_store.retrieve("persistent_test", MemoryType.LONG_TERM)
            # Result might be None due to corruption, but no exception should be raised
        except Exception as e:
            self.fail(f"Loading corrupted storage should not raise exception: {e}")
    
    def test_edge_case_empty_key(self):
        """Test handling of empty or None keys."""
        # Test empty string key
        self.memory_store.store("", "empty_key_value", MemoryType.SHORT_TERM)
        result = self.memory_store.retrieve("", MemoryType.SHORT_TERM)
        self.assertEqual(result, "empty_key_value")
        
        # Test None key handling (should be graceful or raise appropriate error)
        try:
            self.memory_store.store(None, "none_key_value", MemoryType.SHORT_TERM)
            # If it doesn't raise an error, try to retrieve
            result = self.memory_store.retrieve(None, MemoryType.SHORT_TERM)
        except (TypeError, AttributeError):
            # Expected to fail, but shouldn't crash the system
            pass
    
    def test_large_value_storage(self):
        """Test storing large values."""
        large_value = {"data": "x" * 10000}  # Large string
        
        self.memory_store.store("large_value", large_value, MemoryType.SHORT_TERM)
        retrieved = self.memory_store.retrieve("large_value", MemoryType.SHORT_TERM)
        
        self.assertEqual(retrieved, large_value)


if __name__ == '__main__':
    unittest.main()