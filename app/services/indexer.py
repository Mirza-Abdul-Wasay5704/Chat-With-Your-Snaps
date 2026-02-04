"""
Master Index management service for the Memories Retrieval System.

Responsible for:
- Managing the master index (image_id -> metadata + vector IDs)
- CRUD operations on index entries
- Persisting index to disk
- SQLite database operations for metadata storage

The master index is the SOURCE OF TRUTH for the system.
Never re-scan storage to rebuild - always trust the index.
"""
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from app import config
from app.models.schemas import MasterIndexEntry, ImageMetadataDB
from app.utils.logging import get_logger

logger = get_logger(__name__)


# -------------------- SQLITE DATABASE --------------------

def _get_db_connection() -> sqlite3.Connection:
    """Get a SQLite database connection."""
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_connection():
    """Context manager for database connections."""
    conn = _get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """
    Initialize the SQLite database with required tables.
    
    Creates tables if they don't exist.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Main images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                firebase_url TEXT,
                original_date TEXT NOT NULL,
                location TEXT,
                user_id TEXT NOT NULL DEFAULT 'me',
                caption TEXT,
                text_vector_id INTEGER,
                image_vector_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Index on user_id for future multi-user support
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_user_id 
            ON images(user_id)
        """)
        
        # Index on original_date for date-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_date 
            ON images(original_date)
        """)
        
        logger.info("Database initialized")


def save_image_metadata(entry: MasterIndexEntry) -> bool:
    """
    Save or update image metadata in the database.
    
    Args:
        entry: MasterIndexEntry to save.
        
    Returns:
        True if saved successfully.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO images 
            (image_id, firebase_url, original_date, location, user_id,
             caption, text_vector_id, image_vector_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT created_at FROM images WHERE image_id = ?), ?),
                    ?)
        """, (
            entry.image_id,
            entry.firebase_url,
            entry.original_date.isoformat(),
            entry.location,
            entry.user_id,
            entry.caption,
            entry.text_vector_id,
            entry.image_vector_id,
            entry.image_id,  # For COALESCE subquery
            now,  # created_at if new
            now   # updated_at
        ))
        
        return True


def get_image_metadata(image_id: str) -> Optional[MasterIndexEntry]:
    """
    Retrieve image metadata by image_id.
    
    Args:
        image_id: SHA256 hash of the image.
        
    Returns:
        MasterIndexEntry or None if not found.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM images WHERE image_id = ?",
            (image_id,)
        )
        
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return MasterIndexEntry(
            image_id=row["image_id"],
            firebase_url=row["firebase_url"],
            original_date=datetime.fromisoformat(row["original_date"]),
            location=row["location"],
            user_id=row["user_id"],
            caption=row["caption"],
            text_vector_id=row["text_vector_id"],
            image_vector_id=row["image_vector_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )


def get_all_images(user_id: str = "me") -> List[MasterIndexEntry]:
    """
    Retrieve all images for a user.
    
    Args:
        user_id: User identifier.
        
    Returns:
        List of MasterIndexEntry objects.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM images WHERE user_id = ? ORDER BY original_date DESC",
            (user_id,)
        )
        
        entries = []
        for row in cursor.fetchall():
            entries.append(MasterIndexEntry(
                image_id=row["image_id"],
                firebase_url=row["firebase_url"],
                original_date=datetime.fromisoformat(row["original_date"]),
                location=row["location"],
                user_id=row["user_id"],
                caption=row["caption"],
                text_vector_id=row["text_vector_id"],
                image_vector_id=row["image_vector_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        
        return entries


def image_exists(image_id: str) -> bool:
    """
    Check if an image exists in the database.
    
    Args:
        image_id: SHA256 hash of the image.
        
    Returns:
        True if exists.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM images WHERE image_id = ?",
            (image_id,)
        )
        
        return cursor.fetchone() is not None


def delete_image_metadata(image_id: str) -> bool:
    """
    Delete image metadata from the database.
    
    Args:
        image_id: SHA256 hash of the image.
        
    Returns:
        True if deleted.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM images WHERE image_id = ?",
            (image_id,)
        )
        
        return cursor.rowcount > 0


def get_image_count(user_id: str = "me") -> int:
    """
    Get total image count for a user.
    
    Args:
        user_id: User identifier.
        
    Returns:
        Number of images.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM images WHERE user_id = ?",
            (user_id,)
        )
        
        return cursor.fetchone()[0]


# -------------------- JSON MASTER INDEX --------------------

def save_master_index_json() -> None:
    """
    Export the master index to JSON file.
    
    This provides a portable backup of the index.
    The database is the primary storage; this is supplementary.
    """
    entries = get_all_images()
    
    # Convert to dict format: image_id -> entry data
    index_dict = {}
    for entry in entries:
        index_dict[entry.image_id] = entry.model_dump(mode='json')
    
    with open(config.MASTER_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2, default=str)
    
    logger.info(f"Saved master index JSON with {len(index_dict)} entries")


def load_master_index_json() -> Dict[str, Any]:
    """
    Load the master index from JSON file.
    
    Returns:
        Dict mapping image_id to entry data.
    """
    if not config.MASTER_INDEX_PATH.exists():
        return {}
    
    with open(config.MASTER_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def update_vector_ids(
    image_id: str,
    text_vector_id: Optional[int] = None,
    image_vector_id: Optional[int] = None
) -> bool:
    """
    Update FAISS vector IDs for an image.
    
    Args:
        image_id: SHA256 hash of the image.
        text_vector_id: Index in text FAISS index.
        image_vector_id: Index in image FAISS index.
        
    Returns:
        True if updated successfully.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            UPDATE images 
            SET text_vector_id = COALESCE(?, text_vector_id),
                image_vector_id = COALESCE(?, image_vector_id),
                updated_at = ?
            WHERE image_id = ?
        """, (text_vector_id, image_vector_id, now, image_id))
        
        return cursor.rowcount > 0


def update_caption(image_id: str, caption: str) -> bool:
    """
    Update the caption for an image.
    
    Args:
        image_id: SHA256 hash of the image.
        caption: Generated caption text.
        
    Returns:
        True if updated successfully.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            UPDATE images 
            SET caption = ?, updated_at = ?
            WHERE image_id = ?
        """, (caption, now, image_id))
        
        return cursor.rowcount > 0


def get_images_without_captions(limit: int = 100) -> List[MasterIndexEntry]:
    """
    Get images that don't have captions yet.
    
    Args:
        limit: Maximum number to return.
        
    Returns:
        List of MasterIndexEntry objects needing captions.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM images 
            WHERE caption IS NULL 
            ORDER BY created_at 
            LIMIT ?
        """, (limit,))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(MasterIndexEntry(
                image_id=row["image_id"],
                firebase_url=row["firebase_url"],
                original_date=datetime.fromisoformat(row["original_date"]),
                location=row["location"],
                user_id=row["user_id"],
                caption=row["caption"],
                text_vector_id=row["text_vector_id"],
                image_vector_id=row["image_vector_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        
        return entries


def get_images_without_embeddings(limit: int = 100) -> List[MasterIndexEntry]:
    """
    Get images that don't have embeddings yet.
    
    Args:
        limit: Maximum number to return.
        
    Returns:
        List of MasterIndexEntry objects needing embeddings.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM images 
            WHERE text_vector_id IS NULL OR image_vector_id IS NULL
            ORDER BY created_at 
            LIMIT ?
        """, (limit,))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(MasterIndexEntry(
                image_id=row["image_id"],
                firebase_url=row["firebase_url"],
                original_date=datetime.fromisoformat(row["original_date"]),
                location=row["location"],
                user_id=row["user_id"],
                caption=row["caption"],
                text_vector_id=row["text_vector_id"],
                image_vector_id=row["image_vector_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        
        return entries


# -------------------- PHASE 2: EMBEDDING PIPELINE HELPERS --------------------

def get_images_with_captions_without_embeddings(limit: int = 100) -> List[MasterIndexEntry]:
    """
    Get images that have captions but don't have embeddings yet.
    
    This is used by the embedding pipeline to process captioned images.
    
    Args:
        limit: Maximum number to return.
        
    Returns:
        List of MasterIndexEntry objects ready for embedding.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM images 
            WHERE caption IS NOT NULL 
              AND (text_vector_id IS NULL OR image_vector_id IS NULL)
            ORDER BY created_at 
            LIMIT ?
        """, (limit,))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(MasterIndexEntry(
                image_id=row["image_id"],
                firebase_url=row["firebase_url"],
                original_date=datetime.fromisoformat(row["original_date"]),
                location=row["location"],
                user_id=row["user_id"],
                caption=row["caption"],
                text_vector_id=row["text_vector_id"],
                image_vector_id=row["image_vector_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        
        return entries


def get_all_images_with_embeddings(user_id: str = "me") -> List[MasterIndexEntry]:
    """
    Get all images that have both embeddings.
    
    Args:
        user_id: User identifier.
        
    Returns:
        List of fully embedded MasterIndexEntry objects.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM images 
            WHERE user_id = ? 
              AND text_vector_id IS NOT NULL 
              AND image_vector_id IS NOT NULL
            ORDER BY original_date DESC
        """, (user_id,))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(MasterIndexEntry(
                image_id=row["image_id"],
                firebase_url=row["firebase_url"],
                original_date=datetime.fromisoformat(row["original_date"]),
                location=row["location"],
                user_id=row["user_id"],
                caption=row["caption"],
                text_vector_id=row["text_vector_id"],
                image_vector_id=row["image_vector_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        
        return entries


def get_embedding_stats() -> Dict[str, int]:
    """
    Get statistics about embeddings.
    
    Returns:
        Dict with counts of images in various states.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Total images
        cursor.execute("SELECT COUNT(*) FROM images")
        total = cursor.fetchone()[0]
        
        # With captions
        cursor.execute("SELECT COUNT(*) FROM images WHERE caption IS NOT NULL")
        with_captions = cursor.fetchone()[0]
        
        # With text embeddings
        cursor.execute("SELECT COUNT(*) FROM images WHERE text_vector_id IS NOT NULL")
        with_text_emb = cursor.fetchone()[0]
        
        # With image embeddings
        cursor.execute("SELECT COUNT(*) FROM images WHERE image_vector_id IS NOT NULL")
        with_image_emb = cursor.fetchone()[0]
        
        # Fully processed (has both embeddings)
        cursor.execute("""
            SELECT COUNT(*) FROM images 
            WHERE text_vector_id IS NOT NULL AND image_vector_id IS NOT NULL
        """)
        fully_processed = cursor.fetchone()[0]
        
        return {
            "total_images": total,
            "with_captions": with_captions,
            "without_captions": total - with_captions,
            "with_text_embeddings": with_text_emb,
            "with_image_embeddings": with_image_emb,
            "fully_processed": fully_processed,
            "pending_captions": total - with_captions,
            "pending_embeddings": with_captions - fully_processed
        }


def get_image_by_vector_ids(
    text_vector_id: Optional[int] = None,
    image_vector_id: Optional[int] = None
) -> Optional[MasterIndexEntry]:
    """
    Get image metadata by FAISS vector IDs.
    
    Args:
        text_vector_id: Text vector ID in FAISS.
        image_vector_id: Image vector ID in FAISS.
        
    Returns:
        MasterIndexEntry or None if not found.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        if text_vector_id is not None:
            cursor.execute(
                "SELECT * FROM images WHERE text_vector_id = ?",
                (text_vector_id,)
            )
        elif image_vector_id is not None:
            cursor.execute(
                "SELECT * FROM images WHERE image_vector_id = ?",
                (image_vector_id,)
            )
        else:
            return None
        
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return MasterIndexEntry(
            image_id=row["image_id"],
            firebase_url=row["firebase_url"],
            original_date=datetime.fromisoformat(row["original_date"]),
            location=row["location"],
            user_id=row["user_id"],
            caption=row["caption"],
            text_vector_id=row["text_vector_id"],
            image_vector_id=row["image_vector_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )
