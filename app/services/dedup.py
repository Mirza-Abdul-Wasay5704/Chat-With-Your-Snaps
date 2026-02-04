"""
Deduplication service for the Memories Retrieval System.

Responsible for:
- Computing image IDs (SHA256)
- Tracking known image IDs
- Checking for duplicates before storage
"""
from typing import Set, Optional
import json
from pathlib import Path

from app import config
from app.utils.hashing import compute_image_id, is_valid_image_id
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DeduplicationService:
    """
    Service for managing image deduplication.
    
    Uses SHA256 hash of image bytes as the unique identifier.
    Maintains a set of known image IDs to check for duplicates.
    """
    
    def __init__(self):
        """Initialize deduplication service."""
        self._known_ids: Set[str] = set()
        self._load_existing_ids()
    
    def _load_existing_ids(self) -> None:
        """
        Load existing image IDs from the master index.
        
        This ensures we don't re-upload images that are already stored.
        """
        index_path = config.MASTER_INDEX_PATH
        
        if not index_path.exists():
            logger.info("No existing master index found, starting fresh")
            return
        
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            
            # Extract all image_ids from the index
            if isinstance(index_data, dict):
                self._known_ids = set(index_data.keys())
            elif isinstance(index_data, list):
                self._known_ids = {
                    item["image_id"] 
                    for item in index_data 
                    if "image_id" in item
                }
            
            logger.info(f"Loaded {len(self._known_ids)} existing image IDs")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load existing index: {e}")
    
    def get_image_id(self, image_bytes: bytes) -> str:
        """
        Compute the unique ID for image bytes.
        
        Args:
            image_bytes: Raw JPEG bytes of the image.
            
        Returns:
            SHA256 hash string (64 hex characters).
        """
        return compute_image_id(image_bytes)
    
    def is_duplicate(self, image_id: str) -> bool:
        """
        Check if an image ID already exists.
        
        Args:
            image_id: SHA256 hash to check.
            
        Returns:
            True if this image already exists, False otherwise.
        """
        return image_id in self._known_ids
    
    def check_and_register(self, image_bytes: bytes) -> tuple[str, bool]:
        """
        Check if image is duplicate and register if new.
        
        This is the main entry point for deduplication.
        
        Args:
            image_bytes: Raw JPEG bytes of the image.
            
        Returns:
            Tuple of (image_id, is_duplicate).
        """
        image_id = self.get_image_id(image_bytes)
        
        if self.is_duplicate(image_id):
            logger.debug(f"Duplicate detected: {image_id[:16]}...")
            return image_id, True
        
        # Register this ID as known
        self._known_ids.add(image_id)
        return image_id, False
    
    def register_id(self, image_id: str) -> None:
        """
        Manually register an image ID as known.
        
        Args:
            image_id: SHA256 hash to register.
        """
        if not is_valid_image_id(image_id):
            raise ValueError(f"Invalid image_id format: {image_id}")
        
        self._known_ids.add(image_id)
    
    def get_known_count(self) -> int:
        """
        Get the number of known image IDs.
        
        Returns:
            Count of unique images tracked.
        """
        return len(self._known_ids)
    
    def clear(self) -> None:
        """
        Clear all known IDs.
        
        WARNING: Use with caution. This doesn't delete stored images.
        """
        self._known_ids.clear()
        logger.warning("Cleared all known image IDs from memory")


# Singleton instance for the application
_dedup_service: Optional[DeduplicationService] = None


def get_dedup_service() -> DeduplicationService:
    """
    Get the singleton deduplication service instance.
    
    Returns:
        DeduplicationService instance.
    """
    global _dedup_service
    
    if _dedup_service is None:
        _dedup_service = DeduplicationService()
    
    return _dedup_service
