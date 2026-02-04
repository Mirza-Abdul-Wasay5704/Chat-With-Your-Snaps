"""
Downloader service for the Memories Retrieval System.

Responsible for downloading media from Snapchat URLs to temporary storage.
Downloads are performed concurrently for efficiency.
"""
import os
import uuid
import tempfile
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import requests

from app import config
from app.models.schemas import MemoryEntry
from app.utils.logging import get_logger, log_progress

logger = get_logger(__name__)


@dataclass
class DownloadResult:
    """Result of a single download operation."""
    success: bool
    temp_path: Optional[str]  # Path to downloaded file in /tmp
    entry: MemoryEntry  # Original entry
    error: Optional[str] = None


def download_single_media(
    entry: MemoryEntry,
    temp_dir: str,
    timeout: int = None
) -> DownloadResult:
    """
    Download a single media file from Snapchat URL.
    
    Args:
        entry: MemoryEntry with download URL.
        temp_dir: Directory for temporary storage.
        timeout: Request timeout in seconds.
        
    Returns:
        DownloadResult with success status and temp file path.
    """
    if timeout is None:
        timeout = config.DOWNLOAD_TIMEOUT
    
    # Generate unique temp filename (not based on original data)
    temp_filename = f"{uuid.uuid4().hex}.tmp"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        response = requests.get(
            str(entry.download_url),
            timeout=timeout,
            stream=True
        )
        response.raise_for_status()
        
        # Write to temp file
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return DownloadResult(
            success=True,
            temp_path=temp_path,
            entry=entry
        )
        
    except requests.RequestException as e:
        logger.warning(f"Download failed for {entry.date}: {e}")
        return DownloadResult(
            success=False,
            temp_path=None,
            entry=entry,
            error=str(e)
        )
    except IOError as e:
        logger.warning(f"Failed to write temp file: {e}")
        return DownloadResult(
            success=False,
            temp_path=None,
            entry=entry,
            error=str(e)
        )


def download_all_media(
    entries: List[MemoryEntry],
    max_workers: Optional[int] = None
) -> Tuple[List[DownloadResult], List[DownloadResult]]:
    """
    Download all media files concurrently.
    
    Downloads to a temporary directory that should be cleaned up after processing.
    
    Args:
        entries: List of MemoryEntry objects to download.
        max_workers: Maximum concurrent downloads.
        
    Returns:
        Tuple of (successful_downloads, failed_downloads).
    """
    if max_workers is None:
        max_workers = config.MAX_DOWNLOAD_WORKERS
    
    # Ensure temp directory exists
    temp_dir = str(config.TEMP_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"Starting download of {len(entries)} media files...")
    
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_entry = {
            executor.submit(download_single_media, entry, temp_dir): entry
            for entry in entries
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_entry):
            completed += 1
            result = future.result()
            
            if result.success:
                successful.append(result)
            else:
                failed.append(result)
            
            log_progress(logger, completed, len(entries), "Downloading", every_n=50)
    
    logger.info(f"Downloads complete: {len(successful)} successful, {len(failed)} failed")
    
    return successful, failed


def cleanup_temp_files(download_results: List[DownloadResult]) -> int:
    """
    Clean up temporary files after processing.
    
    Args:
        download_results: List of DownloadResult objects.
        
    Returns:
        Number of files cleaned up.
    """
    cleaned = 0
    
    for result in download_results:
        if result.temp_path and os.path.exists(result.temp_path):
            try:
                os.remove(result.temp_path)
                cleaned += 1
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
    
    logger.info(f"Cleaned up {cleaned} temporary files")
    return cleaned


def get_temp_file_size(temp_path: str) -> int:
    """
    Get the size of a temporary file in bytes.
    
    Args:
        temp_path: Path to the temp file.
        
    Returns:
        File size in bytes.
    """
    return os.path.getsize(temp_path)
