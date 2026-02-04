"""
Image processor service for the Memories Retrieval System.

Responsible for:
- Detecting if downloaded media is an image or ZIP
- Extracting main + overlay from Snapchat ZIPs
- Merging layers via alpha compositing
- Producing final JPEG bytes
"""
import io
import zipfile
from typing import Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from app.utils.image_ops import (
    validate_image,
    open_image_from_bytes,
    alpha_composite_overlay,
    image_to_jpeg_bytes,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a downloaded media file."""
    success: bool
    image_bytes: Optional[bytes]  # Final JPEG bytes
    is_zip: bool  # Was this a ZIP file?
    error: Optional[str] = None


def read_file_bytes(file_path: str) -> bytes:
    """
    Read all bytes from a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Raw bytes of the file.
    """
    with open(file_path, "rb") as f:
        return f.read()


def is_zip_file(file_bytes: bytes) -> bool:
    """
    Check if bytes represent a ZIP file.
    
    Args:
        file_bytes: Raw bytes to check.
        
    Returns:
        True if this is a valid ZIP file.
    """
    # Check ZIP magic number
    if len(file_bytes) < 4:
        return False
    
    # ZIP files start with PK\x03\x04
    return file_bytes[:4] == b'PK\x03\x04'


def extract_snapchat_layers(
    zip_bytes: bytes
) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Extract main image and overlay from a Snapchat ZIP.
    
    Snapchat ZIPs contain:
    - A "main" image (the photo)
    - An "overlay" PNG (stickers, text, filters)
    
    Args:
        zip_bytes: Raw bytes of the ZIP file.
        
    Returns:
        Tuple of (main_image_bytes, overlay_image_bytes).
        Either may be None if not found.
    """
    main_bytes = None
    overlay_bytes = None
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
            for filename in z.namelist():
                lower_name = filename.lower()
                
                if "main" in lower_name:
                    main_bytes = z.read(filename)
                elif "overlay" in lower_name:
                    overlay_bytes = z.read(filename)
                    
    except zipfile.BadZipFile as e:
        logger.warning(f"Invalid ZIP file: {e}")
    
    return main_bytes, overlay_bytes


def merge_snapchat_layers(
    main_bytes: bytes,
    overlay_bytes: bytes
) -> bytes:
    """
    Merge Snapchat main image with overlay using alpha compositing.
    
    Args:
        main_bytes: Raw bytes of the main image.
        overlay_bytes: Raw bytes of the overlay PNG.
        
    Returns:
        Final merged image as JPEG bytes.
        
    Raises:
        ValueError: If images cannot be opened or merged.
    """
    try:
        main_image = open_image_from_bytes(main_bytes)
        overlay_image = open_image_from_bytes(overlay_bytes)
        
        # Alpha composite
        merged = alpha_composite_overlay(main_image, overlay_image)
        
        # Convert to JPEG bytes
        jpeg_bytes = image_to_jpeg_bytes(merged)
        
        # Cleanup
        main_image.close()
        overlay_image.close()
        merged.close()
        
        return jpeg_bytes
        
    except Exception as e:
        raise ValueError(f"Failed to merge layers: {e}")


def process_media_file(file_path: str) -> ProcessingResult:
    """
    Process a downloaded media file and produce final JPEG bytes.
    
    Handles both:
    - Direct images: validate and convert to JPEG
    - ZIP files: extract, merge layers, convert to JPEG
    
    Args:
        file_path: Path to the downloaded temp file.
        
    Returns:
        ProcessingResult with success status and final image bytes.
    """
    try:
        file_bytes = read_file_bytes(file_path)
    except IOError as e:
        return ProcessingResult(
            success=False,
            image_bytes=None,
            is_zip=False,
            error=f"Failed to read file: {e}"
        )
    
    # Case 1: It's a ZIP file
    if is_zip_file(file_bytes):
        logger.debug("Detected ZIP file, extracting layers...")
        
        main_bytes, overlay_bytes = extract_snapchat_layers(file_bytes)
        
        if main_bytes and overlay_bytes:
            # Both layers found - merge them
            try:
                merged_bytes = merge_snapchat_layers(main_bytes, overlay_bytes)
                return ProcessingResult(
                    success=True,
                    image_bytes=merged_bytes,
                    is_zip=True
                )
            except ValueError as e:
                return ProcessingResult(
                    success=False,
                    image_bytes=None,
                    is_zip=True,
                    error=str(e)
                )
        
        elif main_bytes:
            # Only main image found - convert to JPEG
            try:
                img = open_image_from_bytes(main_bytes)
                jpeg_bytes = image_to_jpeg_bytes(img)
                img.close()
                return ProcessingResult(
                    success=True,
                    image_bytes=jpeg_bytes,
                    is_zip=True
                )
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    image_bytes=None,
                    is_zip=True,
                    error=f"Failed to process main image from ZIP: {e}"
                )
        
        else:
            return ProcessingResult(
                success=False,
                image_bytes=None,
                is_zip=True,
                error="ZIP file missing main image"
            )
    
    # Case 2: It's a direct image
    if validate_image(file_bytes):
        try:
            img = open_image_from_bytes(file_bytes)
            jpeg_bytes = image_to_jpeg_bytes(img)
            img.close()
            return ProcessingResult(
                success=True,
                image_bytes=jpeg_bytes,
                is_zip=False
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                image_bytes=None,
                is_zip=False,
                error=f"Failed to process image: {e}"
            )
    
    # Case 3: Unknown format
    return ProcessingResult(
        success=False,
        image_bytes=None,
        is_zip=False,
        error="File is neither a valid image nor ZIP"
    )
