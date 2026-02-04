"""
Parser service for the Memories Retrieval System.

Responsible for parsing memories.json from Snapchat exports
and extracting valid image entries.
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Iterator
from pathlib import Path

from app.models.schemas import MemoryEntry, MediaType
from app.utils.logging import get_logger

logger = get_logger(__name__)


def parse_memories_json(json_content: str) -> List[MemoryEntry]:
    """
    Parse the memories.json content and extract valid image entries.
    
    Handles both dict-wrapped and direct list formats from Snapchat exports.
    Only returns entries where:
    - Media Type is "Image"
    - Date is present
    - Media Download Url is present
    
    Args:
        json_content: Raw JSON string from memories.json file.
        
    Returns:
        List of validated MemoryEntry objects for images only.
        
    Raises:
        json.JSONDecodeError: If JSON is malformed.
        ValueError: If no valid entries found.
    """
    data = json.loads(json_content)
    
    # Handle both formats: {"Saved Media": [...]} or [...]
    if isinstance(data, dict):
        # Find the first list value in the dict
        memories_list = None
        for key, value in data.items():
            if isinstance(value, list):
                memories_list = value
                logger.info(f"Found memories under key: '{key}'")
                break
        
        if memories_list is None:
            raise ValueError("No list found in memories.json dict")
    elif isinstance(data, list):
        memories_list = data
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")
    
    logger.info(f"Total entries in JSON: {len(memories_list)}")
    
    valid_entries = []
    skipped_count = 0
    
    for item in memories_list:
        try:
            # Validate required fields exist
            if not all(k in item for k in ["Date", "Media Type", "Media Download Url"]):
                skipped_count += 1
                continue
            
            # Only process images
            if item.get("Media Type") != "Image":
                skipped_count += 1
                continue
            
            entry = MemoryEntry.model_validate(item)
            valid_entries.append(entry)
            
        except Exception as e:
            logger.warning(f"Skipping invalid entry: {e}")
            skipped_count += 1
    
    logger.info(f"Valid image entries: {len(valid_entries)}, Skipped: {skipped_count}")
    
    if not valid_entries:
        raise ValueError("No valid image entries found in memories.json")
    
    return valid_entries


def parse_date_string(date_str: str) -> datetime:
    """
    Parse Snapchat date string format to datetime.
    
    Expected format: "2024-03-15 14:30:45 UTC"
    
    Args:
        date_str: Date string from Snapchat export.
        
    Returns:
        Parsed datetime object (UTC).
        
    Raises:
        ValueError: If date format is not recognized.
    """
    # Try the standard Snapchat format
    formats = [
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unrecognized date format: {date_str}")


def load_memories_from_file(file_path: str) -> List[MemoryEntry]:
    """
    Load and parse memories.json from a file path.
    
    Args:
        file_path: Path to the memories.json file.
        
    Returns:
        List of validated MemoryEntry objects.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"memories.json not found at: {file_path}")
    
    logger.info(f"Loading memories from: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return parse_memories_json(content)


def iter_memory_entries(entries: List[MemoryEntry]) -> Iterator[MemoryEntry]:
    """
    Iterate over memory entries sorted by date (newest first).
    
    This is a generator for memory-efficient processing.
    
    Args:
        entries: List of MemoryEntry objects.
        
    Yields:
        MemoryEntry objects sorted by date descending.
    """
    # Sort by date, newest first
    sorted_entries = sorted(
        entries,
        key=lambda e: parse_date_string(e.date),
        reverse=True
    )
    
    for entry in sorted_entries:
        yield entry
