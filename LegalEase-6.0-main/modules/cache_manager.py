# modules/cache_manager.py
"""
Document processing cache with hash-based storage for performance optimization.
Caches processed documents to avoid reprocessing the same content.
"""

import hashlib
import json
import os
import pickle
import time
import threading
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DocumentCache:
    """
    High-performance document cache with hash-based storage and automatic cleanup.
    """
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 500, max_age_hours: int = 24):
        # Use absolute path relative to the project root
        if not os.path.isabs(cache_dir):
            # Get the project root directory (where app.py is located)
            project_root = Path(__file__).parent.parent
            self.cache_dir = project_root / cache_dir
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.max_age = max_age_hours * 3600  # Convert to seconds
        
        self._lock = threading.RLock()
        self._metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"DocumentCache initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_hash(self, content: str, processing_type: str, **kwargs) -> str:
        """Generate a unique hash for content and processing parameters."""
        # Create a string that includes content, processing type, and parameters
        hash_input = f"{content}{processing_type}{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_hash: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_hash}.cache"
    
    def _periodic_cleanup(self):
        """Periodic cleanup of old and oversized cache entries."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_old_entries()
                self._cleanup_oversized_cache()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _cleanup_old_entries(self):
        """Remove cache entries older than max_age."""
        current_time = time.time()
        removed_count = 0
        
        with self._lock:
            entries_to_remove = []
            
            for cache_hash, metadata in self._metadata.items():
                if current_time - metadata.get('created_at', 0) > self.max_age:
                    entries_to_remove.append(cache_hash)
            
            for cache_hash in entries_to_remove:
                self._remove_cache_entry(cache_hash)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache entries")
    
    def _cleanup_oversized_cache(self):
        """Remove oldest entries if cache exceeds max size."""
        current_size = self._get_cache_size()
        
        if current_size <= self.max_cache_size:
            return
        
        with self._lock:
            # Sort by access time (oldest first)
            sorted_entries = sorted(
                self._metadata.items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )
            
            removed_count = 0
            for cache_hash, _ in sorted_entries:
                if current_size <= self.max_cache_size:
                    break
                
                entry_size = self._metadata[cache_hash].get('size', 0)
                self._remove_cache_entry(cache_hash)
                current_size -= entry_size
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} cache entries due to size limit")
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(metadata.get('size', 0) for metadata in self._metadata.values())
    
    def _remove_cache_entry(self, cache_hash: str):
        """Remove a cache entry from disk and metadata."""
        try:
            cache_path = self._get_cache_path(cache_hash)
            if cache_path.exists():
                cache_path.unlink()
            
            if cache_hash in self._metadata:
                del self._metadata[cache_hash]
            
        except Exception as e:
            logger.error(f"Error removing cache entry {cache_hash}: {e}")
    
    def get(self, content: str, processing_type: str, **kwargs) -> Optional[Any]:
        """
        Get cached result for content and processing type.
        
        Args:
            content: The document content to process
            processing_type: Type of processing (e.g., 'extract_key_info', 'summarize')
            **kwargs: Additional parameters that affect processing
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_hash = self._generate_hash(content, processing_type, **kwargs)
        
        with self._lock:
            if cache_hash not in self._metadata:
                return None
            
            # Check if entry is expired
            metadata = self._metadata[cache_hash]
            if time.time() - metadata.get('created_at', 0) > self.max_age:
                self._remove_cache_entry(cache_hash)
                return None
            
            # Try to load cached result
            try:
                cache_path = self._get_cache_path(cache_hash)
                if not cache_path.exists():
                    # Metadata exists but file doesn't - clean up
                    del self._metadata[cache_hash]
                    return None
                
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                
                # Update last accessed time
                metadata['last_accessed'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                self._save_metadata()
                
                logger.debug(f"Cache hit for {processing_type}: {cache_hash[:8]}")
                return result
                
            except Exception as e:
                logger.error(f"Error loading cache entry {cache_hash}: {e}")
                self._remove_cache_entry(cache_hash)
                return None
    
    def set(self, content: str, processing_type: str, result: Any, **kwargs):
        """
        Cache a processing result.
        
        Args:
            content: The document content that was processed
            processing_type: Type of processing performed
            result: The processing result to cache
            **kwargs: Additional parameters that affected processing
        """
        cache_hash = self._generate_hash(content, processing_type, **kwargs)
        
        try:
            with self._lock:
                # Save result to disk
                cache_path = self._get_cache_path(cache_hash)
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                
                # Update metadata
                file_size = cache_path.stat().st_size
                current_time = time.time()
                
                self._metadata[cache_hash] = {
                    'processing_type': processing_type,
                    'created_at': current_time,
                    'last_accessed': current_time,
                    'access_count': 1,
                    'size': file_size,
                    'content_hash': hashlib.md5(content.encode()).hexdigest()[:16]  # For debugging
                }
                
                self._save_metadata()
                logger.debug(f"Cached result for {processing_type}: {cache_hash[:8]} ({file_size} bytes)")
                
                # Check if we need immediate cleanup
                if self._get_cache_size() > self.max_cache_size:
                    self._cleanup_oversized_cache()
                    
        except Exception as e:
            logger.error(f"Error caching result for {processing_type}: {e}")
    
    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: If provided, only invalidate entries matching this processing type
        """
        with self._lock:
            if pattern is None:
                # Clear entire cache
                for cache_hash in list(self._metadata.keys()):
                    self._remove_cache_entry(cache_hash)
                logger.info("Cleared entire cache")
            else:
                # Clear entries matching pattern
                removed_count = 0
                for cache_hash, metadata in list(self._metadata.items()):
                    if metadata.get('processing_type', '').startswith(pattern):
                        self._remove_cache_entry(cache_hash)
                        removed_count += 1
                logger.info(f"Invalidated {removed_count} cache entries matching '{pattern}'")
            
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = self._get_cache_size()
            entry_count = len(self._metadata)
            
            processing_types = {}
            total_access_count = 0
            
            for metadata in self._metadata.values():
                proc_type = metadata.get('processing_type', 'unknown')
                processing_types[proc_type] = processing_types.get(proc_type, 0) + 1
                total_access_count += metadata.get('access_count', 0)
            
            return {
                'total_entries': entry_count,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_cache_size / (1024 * 1024),
                'processing_types': processing_types,
                'total_access_count': total_access_count,
                'hit_rate': total_access_count / max(entry_count, 1)
            }

# Global cache instance
document_cache = DocumentCache(
    cache_dir="cache/documents",
    max_cache_size_mb=500,
    max_age_hours=24
)

def cached_processing(processing_type: str, **cache_kwargs):
    """
    Decorator for caching processing results.
    
    Usage:
        @cached_processing('extract_key_info', language='en')
        def extract_key_information(text):
            # Processing logic
            return result
    """
    def decorator(func):
        def wrapper(text, *args, **kwargs):
            # Combine decorator kwargs with function kwargs for cache key
            cache_params = {**cache_kwargs, **kwargs}
            
            # Try to get from cache
            cached_result = document_cache.get(text, processing_type, **cache_params)
            if cached_result is not None:
                return cached_result
            
            # Process and cache result
            result = func(text, *args, **kwargs)
            document_cache.set(text, processing_type, result, **cache_params)
            
            return result
        return wrapper
    return decorator
