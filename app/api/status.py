"""
Status API endpoint for the Memories Retrieval System.

Handles:
- Pipeline status queries
- Job progress tracking
- Embedding statistics (Phase 2)
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import StatusResponse, PipelineStatus
from app.api.ingest import get_job_status
from app.services.indexer import get_image_count, get_embedding_stats
from app.services.faiss_store import get_faiss_store
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/status", tags=["status"])


@router.get("/{job_id}", response_model=StatusResponse)
async def get_pipeline_status(job_id: str):
    """
    Get the status of an ingestion job.
    
    Args:
        job_id: The job ID returned from /ingest endpoint.
    """
    job = get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return StatusResponse(
        job_id=job_id,
        status=job.get("status", PipelineStatus.PENDING),
        progress=job.get("progress", 0),
        current_step=job.get("current_step", ""),
        total_images=job.get("total_images", 0),
        processed_images=job.get("processed_images", 0),
        duplicates_skipped=job.get("duplicates_skipped", 0),
        errors=job.get("errors", []),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at")
    )


@router.get("/")
async def get_system_status():
    """
    Get overall system status and statistics.
    """
    try:
        image_count = get_image_count()
        stats = get_embedding_stats()
    except Exception:
        image_count = 0
        stats = {}
    
    # Get FAISS stats
    faiss_store = get_faiss_store()
    text_vectors = faiss_store.get_text_vector_count()
    image_vectors = faiss_store.get_image_vector_count()
    
    return {
        "status": "healthy",
        "total_images": image_count,
        "storage_backend": "mock",  # Will be "firebase" in production
        "embeddings_enabled": True,
        "search_enabled": True,
        "embedding_stats": {
            "with_captions": stats.get("with_captions", 0),
            "pending_captions": stats.get("pending_captions", 0),
            "fully_processed": stats.get("fully_processed", 0),
            "pending_embeddings": stats.get("pending_embeddings", 0),
        },
        "faiss_stats": {
            "text_vectors": text_vectors,
            "image_vectors": image_vectors,
        }
    }
