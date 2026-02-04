"""
Logging utilities for the Memories Retrieval System.

Provides consistent logging setup across all modules.
"""
import logging
import sys
from typing import Optional

from app import config


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Args:
        name: Logger name (typically __name__).
        level: Log level override (uses config default if None).
        
    Returns:
        Configured logger instance.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        level = level or config.LOG_LEVEL
        logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(config.LOG_FORMAT)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    prefix: str = "Progress",
    every_n: int = 10
) -> None:
    """
    Log progress at regular intervals.
    
    Args:
        logger: Logger instance.
        current: Current item number (1-indexed).
        total: Total number of items.
        prefix: Message prefix.
        every_n: Log every N items.
    """
    if current % every_n == 0 or current == total:
        percent = (current / total) * 100
        logger.info(f"{prefix}: {current}/{total} ({percent:.1f}%)")
