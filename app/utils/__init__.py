"""Utility functions for the Memories Retrieval System."""
from app.utils.hashing import compute_image_id, compute_file_hash, is_valid_image_id
from app.utils.image_ops import (
    validate_image,
    open_image_from_bytes,
    convert_to_rgba,
    resize_to_match,
    alpha_composite_overlay,
    image_to_jpeg_bytes,
    get_image_dimensions,
)
from app.utils.logging import get_logger, log_progress

__all__ = [
    # Hashing
    "compute_image_id",
    "compute_file_hash",
    "is_valid_image_id",
    # Image operations
    "validate_image",
    "open_image_from_bytes",
    "convert_to_rgba",
    "resize_to_match",
    "alpha_composite_overlay",
    "image_to_jpeg_bytes",
    "get_image_dimensions",
    # Logging
    "get_logger",
    "log_progress",
]
