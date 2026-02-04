"""
Query API endpoint for the Memories Retrieval System.

Handles:
- Natural language image search
- Hybrid retrieval (text + image similarity)

Phase 2 Implementation:
- CLIP text embedding for queries
- FAISS search on both text and image indices
- Weighted combination: 70% text + 30% image
"""
import time
from typing import List
from fastapi import APIRouter, HTTPException

from app import config
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    ImageResult,
)
from app.services.indexer import get_all_images, get_image_metadata
from app.services.embedder import get_embedder_service
from app.services.faiss_store import get_faiss_store
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["search"])


@router.post("/search", response_model=QueryResponse)
async def search_images(request: QueryRequest):
    """
    Search images using natural language query.
    
    The search process:
    1. Encode query using CLIP text encoder
    2. Search FAISS text index (caption embeddings)
    3. Search FAISS image index (pixel embeddings)  
    4. Combine scores: 70% text + 30% image
    5. Return top-k results sorted by relevance
    """
    logger.info(f"Search query: {request.query}")
    start_time = time.time()
    
    try:
        # Get services
        embedder = get_embedder_service()
        faiss_store = get_faiss_store()
        
        # Check if indices are populated
        text_count = faiss_store.get_text_vector_count()
        image_count = faiss_store.get_image_vector_count()
        
        if text_count == 0 and image_count == 0:
            logger.warning("FAISS indices are empty - run embedding pipeline first")
            return QueryResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time_ms=0.0
            )
        
        # Generate query embedding
        embed_result = embedder.embed_text(request.query)
        
        if not embed_result.success or embed_result.embedding is None:
            logger.error(f"Failed to embed query: {embed_result.error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to embed query: {embed_result.error}"
            )
        
        query_vector = embed_result.embedding
        
        # Perform hybrid search
        search_results = faiss_store.hybrid_search(
            query_vector=query_vector,
            k=request.top_k,
            text_weight=config.TEXT_SIMILARITY_WEIGHT,
            image_weight=config.IMAGE_SIMILARITY_WEIGHT
        )
        
        # Build response with image metadata
        results = []
        for image_id, score in search_results:
            image_meta = get_image_metadata(image_id)
            
            if image_meta and image_meta.firebase_url:
                results.append(ImageResult(
                    image_id=image_meta.image_id,
                    firebase_url=image_meta.firebase_url,
                    original_date=image_meta.original_date,
                    location=image_meta.location,
                    caption=image_meta.caption,
                    score=score
                ))
        
        search_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Search completed: {len(results)} results in {search_time_ms:.2f}ms")
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {e}"
        )


@router.get("/images")
async def list_all_images(
    limit: int = 50,
    offset: int = 0
):
    """
    List all indexed images (paginated).
    
    Useful for browsing without search.
    """
    try:
        all_images = get_all_images()
        
        # Apply pagination
        paginated = all_images[offset:offset + limit]
        
        return {
            "images": [
                {
                    "image_id": img.image_id,
                    "firebase_url": img.firebase_url,
                    "original_date": img.original_date.isoformat(),
                    "location": img.location,
                    "caption": img.caption,
                    "has_embeddings": img.text_vector_id is not None
                }
                for img in paginated
            ],
            "total": len(all_images),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list images: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list images: {e}"
        )


@router.get("/images/{image_id}")
async def get_image_details(image_id: str):
    """
    Get details for a specific image.
    """
    from app.services.indexer import get_image_metadata
    
    image = get_image_metadata(image_id)
    
    if not image:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found"
        )
    
    return {
        "image_id": image.image_id,
        "firebase_url": image.firebase_url,
        "original_date": image.original_date.isoformat(),
        "location": image.location,
        "caption": image.caption,
        "has_embeddings": image.text_vector_id is not None,
        "created_at": image.created_at.isoformat(),
        "updated_at": image.updated_at.isoformat()
    }
