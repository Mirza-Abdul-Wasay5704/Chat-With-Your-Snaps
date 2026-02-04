"""
Configuration module for the Memories Retrieval System.

Contains all paths, model names, and constants used throughout the application.
Environment variables can override defaults for production deployment.
"""
import os
from pathlib import Path

# -------------------- BASE PATHS --------------------
# Root directory for all persistent data (not images - those go to Firebase)
BASE_DIR = Path(os.getenv("MEMORIES_BASE_DIR", "./data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Temporary directory for processing (downloads, merging, etc.)
TEMP_DIR = Path(os.getenv("MEMORIES_TEMP_DIR", "/tmp/memories_processing"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- DATABASE PATHS --------------------
# SQLite database for metadata + captions
DB_PATH = BASE_DIR / "db.sqlite"

# Master index mapping image_id -> metadata + vector IDs
MASTER_INDEX_PATH = BASE_DIR / "master_index.json"

# FAISS index directory
FAISS_DIR = BASE_DIR / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

FAISS_TEXT_INDEX_PATH = FAISS_DIR / "text.index"
FAISS_IMAGE_INDEX_PATH = FAISS_DIR / "image.index"

# -------------------- FIREBASE CONFIG --------------------
# Firebase Storage bucket name (without gs:// prefix)
FIREBASE_BUCKET = os.getenv("FIREBASE_BUCKET", "chat-with-snap.firebasestorage.app")

# Firebase service account key path
FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH", 
    str(Path(__file__).parent / "chat-with-snap-b9b2bd237e62.json")
)

# Firebase storage path format for images
FIREBASE_IMAGE_PATH_TEMPLATE = "images/{image_id}.jpg"

# -------------------- USER CONFIG --------------------
# Current development phase: single user, hardcoded
DEFAULT_USER_ID = "me"

# -------------------- PROCESSING CONFIG --------------------
# Maximum concurrent downloads
MAX_DOWNLOAD_WORKERS = 8

# Download timeout in seconds
DOWNLOAD_TIMEOUT = 30

# Image quality for JPEG compression (1-100)
JPEG_QUALITY = 95

# -------------------- MODEL CONFIG (FOR LATER PHASES) --------------------
# CLIP model for embeddings
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Florence-2 model for captions
FLORENCE_MODEL_NAME = "microsoft/Florence-2-large"

# -------------------- SEARCH CONFIG (FOR LATER PHASES) --------------------
# Weight for text similarity in hybrid search
TEXT_SIMILARITY_WEIGHT = 0.7

# Weight for image similarity in hybrid search
IMAGE_SIMILARITY_WEIGHT = 0.3

# Default number of results to return
DEFAULT_TOP_K = 10

# -------------------- LOGGING CONFIG --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
