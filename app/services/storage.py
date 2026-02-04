"""
Firebase Storage service for the Memories Retrieval System.

Responsible for:
- Uploading images to Firebase Storage
- Generating access URLs
- Managing storage paths
"""
import os
from typing import Optional
from pathlib import Path

from app import config
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Firebase SDK imports - lazy loaded to allow running without Firebase
_firebase_initialized = False
_bucket = None


def _init_firebase() -> bool:
    """
    Initialize Firebase connection.
    
    Returns:
        True if initialization successful, False otherwise.
    """
    global _firebase_initialized, _bucket
    
    if _firebase_initialized:
        return _bucket is not None
    
    try:
        import firebase_admin
        from firebase_admin import credentials, storage
        
        cred_path = config.FIREBASE_CREDENTIALS_PATH
        
        if not os.path.exists(cred_path):
            logger.error(f"Firebase credentials not found at: {cred_path}")
            _firebase_initialized = True
            return False
        
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': config.FIREBASE_BUCKET
        })
        
        _bucket = storage.bucket()
        _firebase_initialized = True
        
        logger.info(f"Firebase initialized with bucket: {config.FIREBASE_BUCKET}")
        return True
        
    except ImportError:
        logger.error("firebase-admin package not installed")
        _firebase_initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        _firebase_initialized = True
        return False


def get_storage_path(image_id: str) -> str:
    """
    Generate the Firebase Storage path for an image.
    
    Args:
        image_id: SHA256 hash of the image.
        
    Returns:
        Storage path string (e.g., "images/abc123...def.jpg").
    """
    return config.FIREBASE_IMAGE_PATH_TEMPLATE.format(image_id=image_id)


def upload_image(
    image_bytes: bytes,
    image_id: str,
    content_type: str = "image/jpeg"
) -> Optional[str]:
    """
    Upload an image to Firebase Storage.
    
    Args:
        image_bytes: Raw JPEG bytes of the image.
        image_id: SHA256 hash (used as filename).
        content_type: MIME type of the image.
        
    Returns:
        Public URL of the uploaded image, or None if upload failed.
    """
    if not _init_firebase():
        logger.error("Firebase not available, cannot upload")
        return None
    
    storage_path = get_storage_path(image_id)
    
    try:
        blob = _bucket.blob(storage_path)
        
        blob.upload_from_string(
            image_bytes,
            content_type=content_type
        )
        
        # Make the blob publicly accessible
        blob.make_public()
        
        public_url = blob.public_url
        logger.debug(f"Uploaded image: {image_id[:16]}... -> {storage_path}")
        
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to upload image {image_id[:16]}...: {e}")
        return None


def check_image_exists(image_id: str) -> bool:
    """
    Check if an image already exists in Firebase Storage.
    
    Note: This should NOT be used for deduplication.
    Use the master index instead. This is only for verification.
    
    Args:
        image_id: SHA256 hash of the image.
        
    Returns:
        True if image exists in storage.
    """
    if not _init_firebase():
        return False
    
    storage_path = get_storage_path(image_id)
    
    try:
        blob = _bucket.blob(storage_path)
        return blob.exists()
    except Exception as e:
        logger.warning(f"Failed to check image existence: {e}")
        return False


def delete_image(image_id: str) -> bool:
    """
    Delete an image from Firebase Storage.
    
    Args:
        image_id: SHA256 hash of the image.
        
    Returns:
        True if deletion successful.
    """
    if not _init_firebase():
        return False
    
    storage_path = get_storage_path(image_id)
    
    try:
        blob = _bucket.blob(storage_path)
        blob.delete()
        logger.info(f"Deleted image: {image_id[:16]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to delete image {image_id[:16]}...: {e}")
        return False


def get_signed_url(image_id: str, expiration_minutes: int = 60) -> Optional[str]:
    """
    Generate a signed URL for private access to an image.
    
    Args:
        image_id: SHA256 hash of the image.
        expiration_minutes: URL validity period.
        
    Returns:
        Signed URL string, or None if generation failed.
    """
    if not _init_firebase():
        return None
    
    from datetime import timedelta
    
    storage_path = get_storage_path(image_id)
    
    try:
        blob = _bucket.blob(storage_path)
        
        url = blob.generate_signed_url(
            expiration=timedelta(minutes=expiration_minutes),
            method='GET'
        )
        
        return url
        
    except Exception as e:
        logger.error(f"Failed to generate signed URL: {e}")
        return None


class MockStorageService:
    """
    Mock storage service for local development without Firebase.
    
    Stores images in a local directory and returns file:// URLs.
    """
    
    def __init__(self, local_dir: str = None):
        """
        Initialize mock storage.
        
        Args:
            local_dir: Directory for storing images locally.
        """
        if local_dir is None:
            local_dir = str(config.BASE_DIR / "mock_storage")
        
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Using MOCK storage service (local files)")
    
    def upload_image(
        self,
        image_bytes: bytes,
        image_id: str
    ) -> str:
        """Upload image to local directory."""
        file_path = self.local_dir / f"{image_id}.jpg"
        
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        return f"file://{file_path.absolute()}"
    
    def check_image_exists(self, image_id: str) -> bool:
        """Check if image exists locally."""
        file_path = self.local_dir / f"{image_id}.jpg"
        return file_path.exists()
    
    def delete_image(self, image_id: str) -> bool:
        """Delete image from local directory."""
        file_path = self.local_dir / f"{image_id}.jpg"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


# For development, can use mock storage
_mock_storage: Optional[MockStorageService] = None


def get_mock_storage() -> MockStorageService:
    """Get mock storage service for local development."""
    global _mock_storage
    
    if _mock_storage is None:
        _mock_storage = MockStorageService()
    
    return _mock_storage
