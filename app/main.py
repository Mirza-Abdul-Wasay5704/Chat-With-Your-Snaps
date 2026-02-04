"""
Main FastAPI application for the Memories Retrieval System.

This is the entry point for the application.
Run with: uvicorn app.main:app --reload
"""
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app import config
from app.api import ingest_router, status_router, query_router, embeddings_router
from app.services.indexer import init_database
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Runs on startup and shutdown.
    """
    # Startup
    logger.info("Starting Memories Retrieval System...")
    logger.info(f"Data directory: {config.BASE_DIR}")
    logger.info(f"Temp directory: {config.TEMP_DIR}")
    
    # Initialize database
    init_database()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Memories Retrieval System...")


# Create FastAPI application
app = FastAPI(
    title="Memories Retrieval System",
    description="""
    A multimodal image retrieval system for Snapchat Memories.
    
    ## Features (Phase 1)
    - Upload memories.json from Snapchat export
    - Download and process media (images + ZIP overlays)
    - Deduplicate images by content (SHA256)
    - Store images in Firebase Storage (or mock local storage)
    - Maintain master index with metadata
    
    ## Coming Soon (Phase 2)
    - Caption generation with Florence-2
    - CLIP embeddings for text and images
    - Natural language search
    - Hybrid retrieval (70% text + 30% image similarity)
    """,
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest_router)
app.include_router(status_router)
app.include_router(query_router)
app.include_router(embeddings_router)

# Mount static files for serving images from local storage
app.mount("/images", StaticFiles(directory=config.BASE_DIR / "images", check_dir=False), name="images")

# Serve static frontend files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    """Detailed health check."""
    from app.services.indexer import get_image_count
    
    try:
        image_count = get_image_count()
        db_status = "connected"
    except Exception as e:
        image_count = 0
        db_status = f"error: {e}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "total_images": image_count,
        "config": {
            "base_dir": str(config.BASE_DIR),
            "firebase_bucket": config.FIREBASE_BUCKET,
            "max_download_workers": config.MAX_DOWNLOAD_WORKERS,
        }
    }
