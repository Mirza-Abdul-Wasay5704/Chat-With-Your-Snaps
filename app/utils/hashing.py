"""
Hashing utilities for the Memories Retrieval System.

Provides cryptographic hashing functions for image identity.
Image identity is determined ONLY by SHA256 hash of image bytes.
"""
import hashlib
from typing import Union


def compute_image_id(image_bytes: Union[bytes, bytearray]) -> str:
    """
    Compute the unique image ID from raw image bytes.
    
    This is the ONLY way to identify images in the system.
    Never use filenames, timestamps, or index numbers for identity.
    
    Args:
        image_bytes: Raw bytes of the final processed image (JPEG).
        
    Returns:
        SHA256 hash as a lowercase hexadecimal string (64 characters).
        
    Raises:
        ValueError: If image_bytes is empty.
        
    Example:
        >>> image_id = compute_image_id(jpeg_bytes)
        >>> print(image_id)
        'a3f2b8c9d4e5f6...'
    """
    if not image_bytes:
        raise ValueError("Cannot compute image_id: image_bytes is empty")
    
    return hashlib.sha256(image_bytes).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file on disk.
    
    Reads file in chunks to handle large files efficiently.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        SHA256 hash as a lowercase hexadecimal string.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        IOError: If file can't be read.
    """
    sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def is_valid_image_id(image_id: str) -> bool:
    """
    Validate that a string is a valid image ID (SHA256 hex).
    
    Args:
        image_id: String to validate.
        
    Returns:
        True if valid SHA256 hex string, False otherwise.
    """
    if not isinstance(image_id, str):
        return False
    
    if len(image_id) != 64:
        return False
    
    try:
        int(image_id, 16)
        return True
    except ValueError:
        return False
