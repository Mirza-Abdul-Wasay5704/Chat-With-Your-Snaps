"""
Image operations utilities for the Memories Retrieval System.

Provides low-level image manipulation functions:
- Opening and validating images
- Alpha compositing (overlay merging)
- Format conversion and compression
"""
import io
from typing import Tuple, Optional
from PIL import Image, UnidentifiedImageError

from app import config


def validate_image(image_bytes: bytes) -> bool:
    """
    Check if bytes represent a valid image.
    
    Args:
        image_bytes: Raw bytes to validate.
        
    Returns:
        True if valid image, False otherwise.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Open a PIL Image from bytes.
    
    Args:
        image_bytes: Raw image bytes.
        
    Returns:
        PIL Image object (caller must close).
        
    Raises:
        UnidentifiedImageError: If bytes are not a valid image.
    """
    return Image.open(io.BytesIO(image_bytes))


def convert_to_rgba(image: Image.Image) -> Image.Image:
    """
    Convert image to RGBA mode for alpha compositing.
    
    Args:
        image: PIL Image in any mode.
        
    Returns:
        New PIL Image in RGBA mode.
    """
    return image.convert("RGBA")


def resize_to_match(source: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image to match target dimensions.
    
    Uses LANCZOS resampling for high-quality results.
    
    Args:
        source: Image to resize.
        target_size: (width, height) tuple.
        
    Returns:
        Resized image.
    """
    if source.size == target_size:
        return source
    
    return source.resize(target_size, Image.Resampling.LANCZOS)


def alpha_composite_overlay(
    main_image: Image.Image, 
    overlay_image: Image.Image
) -> Image.Image:
    """
    Merge main image with transparent overlay using alpha compositing.
    
    This is used to reconstruct Snapchat images from ZIP files
    that contain separate main and overlay layers.
    
    Args:
        main_image: Base image (will be converted to RGBA).
        overlay_image: Overlay with transparency (will be converted to RGBA).
        
    Returns:
        Composited image in RGBA mode.
    """
    # Convert both to RGBA
    main_rgba = convert_to_rgba(main_image)
    overlay_rgba = convert_to_rgba(overlay_image)
    
    # Resize overlay if dimensions don't match
    if overlay_rgba.size != main_rgba.size:
        overlay_rgba = resize_to_match(overlay_rgba, main_rgba.size)
    
    # Composite: overlay on top of main
    return Image.alpha_composite(main_rgba, overlay_rgba)


def image_to_jpeg_bytes(
    image: Image.Image, 
    quality: Optional[int] = None
) -> bytes:
    """
    Convert PIL Image to JPEG bytes.
    
    Args:
        image: PIL Image in any mode.
        quality: JPEG quality (1-100). Uses config default if None.
        
    Returns:
        JPEG image as bytes.
    """
    if quality is None:
        quality = config.JPEG_QUALITY
    
    # Convert to RGB (JPEG doesn't support alpha)
    rgb_image = image.convert("RGB")
    
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    return buffer.read()


def get_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """
    Get width and height of an image from bytes.
    
    Args:
        image_bytes: Raw image bytes.
        
    Returns:
        (width, height) tuple.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.size
