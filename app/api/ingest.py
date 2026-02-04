"""
Ingestion API endpoint for the Memories Retrieval System.

Handles:
- Upload of memories.json
- Triggering the ingestion pipeline
- Background processing
"""
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app import config
from app.models.schemas import (
    IngestResponse, 
    PipelineStatus,
    MasterIndexEntry,
)
from app.services.parser import parse_memories_json, parse_date_string
from app.services.downloader import download_all_media, cleanup_temp_files
from app.services.image_processor import process_media_file
from app.services.dedup import get_dedup_service
from app.services.storage import upload_image, get_mock_storage
from app.services.indexer import (
    init_database,
    save_image_metadata,
    save_master_index_json,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# In-memory job tracking (for single-user, replace with Redis for multi-user)
_jobs: Dict[str, Dict[str, Any]] = {}


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of a job."""
    return _jobs.get(job_id, {})


def update_job_status(
    job_id: str,
    status: PipelineStatus,
    current_step: str = "",
    progress: float = 0,
    **kwargs
) -> None:
    """Update job status."""
    if job_id not in _jobs:
        _jobs[job_id] = {
            "started_at": datetime.utcnow(),
            "errors": []
        }
    
    _jobs[job_id].update({
        "status": status,
        "current_step": current_step,
        "progress": progress,
        **kwargs
    })


async def run_ingestion_pipeline(
    job_id: str,
    json_content: str,
    use_mock_storage: bool = True
) -> None:
    """
    Run the full ingestion pipeline.
    
    This is executed as a background task.
    
    Args:
        job_id: Unique job identifier.
        json_content: Raw memories.json content.
        use_mock_storage: Use local mock storage instead of Firebase.
    """
    try:
        # Initialize database
        init_database()
        
        # Step 1: Parse JSON
        update_job_status(
            job_id, 
            PipelineStatus.PARSING, 
            "Parsing memories.json",
            5
        )
        
        try:
            entries = parse_memories_json(json_content)
            update_job_status(
                job_id,
                PipelineStatus.PARSING,
                f"Found {len(entries)} image entries",
                10,
                total_images=len(entries)
            )
        except Exception as e:
            update_job_status(
                job_id,
                PipelineStatus.FAILED,
                f"Parse error: {e}",
                errors=[str(e)]
            )
            return
        
        # Step 2: Download media
        update_job_status(
            job_id,
            PipelineStatus.DOWNLOADING,
            "Downloading media files",
            15
        )
        
        successful_downloads, failed_downloads = download_all_media(entries)
        
        update_job_status(
            job_id,
            PipelineStatus.DOWNLOADING,
            f"Downloaded {len(successful_downloads)} files",
            40,
            errors=[r.error for r in failed_downloads if r.error]
        )
        
        # Step 3: Process images (ZIP extraction, merging)
        update_job_status(
            job_id,
            PipelineStatus.PROCESSING,
            "Processing images",
            45
        )
        
        dedup_service = get_dedup_service()
        storage = get_mock_storage() if use_mock_storage else None
        
        processed = 0
        duplicates = 0
        uploaded = 0
        errors = []
        
        for download_result in successful_downloads:
            try:
                # Process the downloaded file
                proc_result = process_media_file(download_result.temp_path)
                
                if not proc_result.success:
                    errors.append(f"Processing failed: {proc_result.error}")
                    continue
                
                # Deduplication check
                image_id, is_duplicate = dedup_service.check_and_register(
                    proc_result.image_bytes
                )
                
                if is_duplicate:
                    duplicates += 1
                    continue
                
                # Upload to storage
                if use_mock_storage:
                    firebase_url = storage.upload_image(
                        proc_result.image_bytes,
                        image_id
                    )
                else:
                    firebase_url = upload_image(
                        proc_result.image_bytes,
                        image_id
                    )
                
                if not firebase_url:
                    errors.append(f"Upload failed for {image_id[:16]}...")
                    continue
                
                # Save to database
                entry = MasterIndexEntry(
                    image_id=image_id,
                    firebase_url=firebase_url,
                    original_date=parse_date_string(download_result.entry.date),
                    location=download_result.entry.location,
                    user_id=config.DEFAULT_USER_ID
                )
                
                save_image_metadata(entry)
                uploaded += 1
                processed += 1
                
                # Update progress
                progress = 45 + (processed / len(successful_downloads)) * 45
                update_job_status(
                    job_id,
                    PipelineStatus.PROCESSING,
                    f"Processed {processed}/{len(successful_downloads)}",
                    progress,
                    processed_images=processed,
                    duplicates_skipped=duplicates
                )
                
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Error processing image: {e}")
        
        # Step 4: Cleanup temp files
        update_job_status(
            job_id,
            PipelineStatus.INDEXING,
            "Cleaning up and saving index",
            92
        )
        
        cleanup_temp_files(successful_downloads)
        
        # Save master index JSON
        save_master_index_json()
        
        # Done!
        update_job_status(
            job_id,
            PipelineStatus.COMPLETED,
            "Ingestion complete",
            100,
            processed_images=uploaded,
            duplicates_skipped=duplicates,
            errors=errors,
            completed_at=datetime.utcnow()
        )
        
        logger.info(
            f"Ingestion complete: {uploaded} uploaded, "
            f"{duplicates} duplicates, {len(errors)} errors"
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        update_job_status(
            job_id,
            PipelineStatus.FAILED,
            str(e),
            errors=[str(e)]
        )


@router.post("/", response_model=IngestResponse)
async def ingest_memories(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The memories.json file from Snapchat export")
):
    """
    Upload memories.json and start the ingestion pipeline.
    
    The pipeline runs in the background. Use /status/{job_id} to check progress.
    """
    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="File must be a JSON file"
        )
    
    # Read file content
    try:
        content = await file.read()
        json_content = content.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read file: {e}"
        )
    
    # Generate job ID
    job_id = uuid.uuid4().hex
    
    # Initialize job status
    update_job_status(
        job_id,
        PipelineStatus.PENDING,
        "Queued for processing"
    )
    
    # Start background processing
    background_tasks.add_task(
        run_ingestion_pipeline,
        job_id,
        json_content,
        use_mock_storage=True  # Use mock for development
    )
    
    return IngestResponse(
        job_id=job_id,
        status=PipelineStatus.PENDING,
        message="Ingestion started. Use /status/{job_id} to check progress."
    )


@router.post("/test-parse")
async def test_parse_memories(
    file: UploadFile = File(..., description="The memories.json file to test parsing")
):
    """
    Test parsing memories.json without running the full pipeline.
    
    Useful for validating the file format before full ingestion.
    """
    try:
        content = await file.read()
        json_content = content.decode("utf-8")
        
        entries = parse_memories_json(json_content)
        
        return {
            "success": True,
            "total_entries": len(entries),
            "sample_entries": [
                {
                    "date": e.date,
                    "media_type": e.media_type,
                    "has_location": e.location is not None
                }
                for e in entries[:5]
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Parse error: {e}"
        )
