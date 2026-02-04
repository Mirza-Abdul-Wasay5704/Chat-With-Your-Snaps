"""
Embeddings API endpoint for the Memories Retrieval System.

Handles:
- Triggering the embedding pipeline (captioning + CLIP embeddings)
- Processing images in the background
- Resumable pipeline execution

Phase 2 Implementation.
"""
import uuid
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app import config
from app.models.schemas import PipelineStatus
from app.services.captioner import get_captioner_service
from app.services.embedder import get_embedder_service
from app.services.faiss_store import get_faiss_store
from app.services.indexer import (
    get_images_without_captions,
    get_images_with_captions_without_embeddings,
    update_caption,
    update_vector_ids,
    save_master_index_json,
    get_embedding_stats,
)
from app.services.storage import get_mock_storage
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# Job tracking
_embedding_jobs: Dict[str, Dict[str, Any]] = {}


def get_embedding_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of an embedding job."""
    return _embedding_jobs.get(job_id, {})


def update_embedding_job(job_id: str, **kwargs) -> None:
    """Update embedding job status."""
    if job_id not in _embedding_jobs:
        _embedding_jobs[job_id] = {"started_at": datetime.utcnow()}
    _embedding_jobs[job_id].update(kwargs)


def _load_image_bytes(firebase_url: str) -> Optional[bytes]:
    """
    Load image bytes from URL (Firebase or local mock).
    
    Args:
        firebase_url: URL of the image.
        
    Returns:
        Image bytes or None if failed.
    """
    try:
        if firebase_url.startswith("file://"):
            # Local mock storage
            file_path = firebase_url[7:]  # Remove file:// prefix
            with open(file_path, "rb") as f:
                return f.read()
        else:
            # HTTP URL (Firebase)
            response = requests.get(firebase_url, timeout=30)
            response.raise_for_status()
            return response.content
    except Exception as e:
        logger.error(f"Failed to load image from {firebase_url}: {e}")
        return None


async def run_captioning_pipeline(
    job_id: str,
    batch_size: int = 10
) -> None:
    """
    Run the captioning pipeline for images without captions.
    
    Args:
        job_id: Job identifier.
        batch_size: Number of images to process per batch.
    """
    captioner = get_captioner_service()
    
    update_embedding_job(
        job_id,
        status=PipelineStatus.PROCESSING,
        current_step="Loading captioning model",
        progress=5
    )
    
    # Load model
    if not captioner.load_model():
        update_embedding_job(
            job_id,
            status=PipelineStatus.FAILED,
            current_step="Failed to load Florence-2 model",
            error="Model loading failed"
        )
        return
    
    # Get images without captions
    images = get_images_without_captions(limit=1000)
    total = len(images)
    
    if total == 0:
        update_embedding_job(
            job_id,
            status=PipelineStatus.COMPLETED,
            current_step="No images need captioning",
            progress=100,
            captioned=0
        )
        return
    
    logger.info(f"Captioning {total} images...")
    
    captioned = 0
    errors = []
    
    for idx, image in enumerate(images):
        try:
            # Load image bytes
            image_bytes = _load_image_bytes(image.firebase_url)
            if image_bytes is None:
                errors.append(f"Failed to load {image.image_id[:16]}...")
                continue
            
            # Generate caption
            result = captioner.generate_caption(image_bytes, image.image_id)
            
            if result.success and result.caption:
                # Save caption to database
                update_caption(image.image_id, result.caption)
                captioned += 1
            else:
                errors.append(f"Caption failed: {result.error}")
            
            # Update progress
            progress = 10 + (idx / total) * 85
            update_embedding_job(
                job_id,
                status=PipelineStatus.PROCESSING,
                current_step=f"Captioning image {idx + 1}/{total}",
                progress=progress,
                captioned=captioned,
                total=total
            )
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Error captioning {image.image_id[:16]}...: {e}")
    
    # Unload model to free memory
    captioner.unload_model()
    
    # Save master index
    save_master_index_json()
    
    update_embedding_job(
        job_id,
        status=PipelineStatus.COMPLETED,
        current_step="Captioning complete",
        progress=100,
        captioned=captioned,
        total=total,
        errors=errors[:10],  # Limit errors
        completed_at=datetime.utcnow()
    )
    
    logger.info(f"Captioning complete: {captioned}/{total} images")


async def run_embedding_pipeline(
    job_id: str,
    batch_size: int = 8
) -> None:
    """
    Run the embedding pipeline for images with captions but no embeddings.
    
    Args:
        job_id: Job identifier.
        batch_size: Number of images to process per batch.
    """
    embedder = get_embedder_service()
    faiss_store = get_faiss_store()
    
    update_embedding_job(
        job_id,
        status=PipelineStatus.PROCESSING,
        current_step="Loading embedding model",
        progress=5
    )
    
    # Load model
    if not embedder.load_model():
        update_embedding_job(
            job_id,
            status=PipelineStatus.FAILED,
            current_step="Failed to load CLIP model",
            error="Model loading failed"
        )
        return
    
    # Ensure FAISS indices are loaded
    if not faiss_store.load_indices():
        update_embedding_job(
            job_id,
            status=PipelineStatus.FAILED,
            current_step="Failed to load FAISS indices",
            error="FAISS loading failed"
        )
        return
    
    # Get images needing embeddings
    images = get_images_with_captions_without_embeddings(limit=1000)
    total = len(images)
    
    if total == 0:
        update_embedding_job(
            job_id,
            status=PipelineStatus.COMPLETED,
            current_step="No images need embedding",
            progress=100,
            embedded=0
        )
        return
    
    logger.info(f"Embedding {total} images...")
    
    embedded = 0
    errors = []
    
    for idx, image in enumerate(images):
        try:
            # Skip if no caption
            if not image.caption:
                continue
            
            # Load image bytes
            image_bytes = _load_image_bytes(image.firebase_url)
            if image_bytes is None:
                errors.append(f"Failed to load {image.image_id[:16]}...")
                continue
            
            # Generate text embedding from caption
            text_result = embedder.embed_text(image.caption)
            
            # Generate image embedding
            image_result = embedder.embed_image(image_bytes)
            
            if text_result.success and image_result.success:
                # Add to FAISS indices
                text_vector_id = faiss_store.add_text_vector(
                    text_result.embedding,
                    image.image_id
                )
                image_vector_id = faiss_store.add_image_vector(
                    image_result.embedding,
                    image.image_id
                )
                
                # Update database
                if text_vector_id is not None and image_vector_id is not None:
                    update_vector_ids(
                        image.image_id,
                        text_vector_id=text_vector_id,
                        image_vector_id=image_vector_id
                    )
                    embedded += 1
            else:
                if not text_result.success:
                    errors.append(f"Text embed failed: {text_result.error}")
                if not image_result.success:
                    errors.append(f"Image embed failed: {image_result.error}")
            
            # Update progress
            progress = 10 + (idx / total) * 80
            update_embedding_job(
                job_id,
                status=PipelineStatus.PROCESSING,
                current_step=f"Embedding image {idx + 1}/{total}",
                progress=progress,
                embedded=embedded,
                total=total
            )
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Error embedding {image.image_id[:16]}...: {e}")
    
    # Save FAISS indices
    update_embedding_job(
        job_id,
        status=PipelineStatus.INDEXING,
        current_step="Saving FAISS indices",
        progress=92
    )
    
    faiss_store.save_indices()
    
    # Save master index
    save_master_index_json()
    
    # Unload model
    embedder.unload_model()
    
    update_embedding_job(
        job_id,
        status=PipelineStatus.COMPLETED,
        current_step="Embedding complete",
        progress=100,
        embedded=embedded,
        total=total,
        errors=errors[:10],
        completed_at=datetime.utcnow()
    )
    
    logger.info(f"Embedding complete: {embedded}/{total} images")


async def run_full_pipeline(job_id: str) -> None:
    """
    Run the full embedding pipeline (captioning + embedding).
    
    Args:
        job_id: Job identifier.
    """
    # Phase 1: Captioning
    update_embedding_job(
        job_id,
        status=PipelineStatus.PROCESSING,
        current_step="Starting captioning phase",
        progress=0,
        phase="captioning"
    )
    
    await run_captioning_pipeline(f"{job_id}_caption")
    
    # Phase 2: Embedding
    update_embedding_job(
        job_id,
        status=PipelineStatus.PROCESSING,
        current_step="Starting embedding phase",
        progress=50,
        phase="embedding"
    )
    
    await run_embedding_pipeline(f"{job_id}_embed")
    
    # Final status
    caption_job = get_embedding_job_status(f"{job_id}_caption")
    embed_job = get_embedding_job_status(f"{job_id}_embed")
    
    update_embedding_job(
        job_id,
        status=PipelineStatus.COMPLETED,
        current_step="Full pipeline complete",
        progress=100,
        captioned=caption_job.get("captioned", 0),
        embedded=embed_job.get("embedded", 0),
        completed_at=datetime.utcnow()
    )


@router.post("/process")
async def process_embeddings(
    background_tasks: BackgroundTasks,
    mode: str = "full"
):
    """
    Trigger the embedding pipeline.
    
    Modes:
    - "full": Run captioning + embedding (default)
    - "caption": Run captioning only
    - "embed": Run embedding only (requires existing captions)
    
    The pipeline runs in the background. Use /embeddings/status/{job_id} to check progress.
    """
    job_id = uuid.uuid4().hex
    
    update_embedding_job(
        job_id,
        status=PipelineStatus.PENDING,
        current_step="Queued for processing",
        mode=mode
    )
    
    if mode == "full":
        background_tasks.add_task(run_full_pipeline, job_id)
    elif mode == "caption":
        background_tasks.add_task(run_captioning_pipeline, job_id)
    elif mode == "embed":
        background_tasks.add_task(run_embedding_pipeline, job_id)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Use 'full', 'caption', or 'embed'"
        )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,
        "message": f"Embedding pipeline started. Use /embeddings/status/{job_id} to check progress."
    }


@router.get("/status/{job_id}")
async def get_pipeline_status(job_id: str):
    """
    Get the status of an embedding pipeline job.
    """
    job = get_embedding_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return {
        "job_id": job_id,
        **job
    }


@router.get("/stats")
async def get_stats():
    """
    Get embedding statistics.
    """
    try:
        stats = get_embedding_stats()
        faiss_store = get_faiss_store()
        
        return {
            "database": stats,
            "faiss": {
                "text_vectors": faiss_store.get_text_vector_count(),
                "image_vectors": faiss_store.get_image_vector_count(),
            }
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {e}"
        )
