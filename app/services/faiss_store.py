"""
FAISS vector store service for the Memories Retrieval System.

Responsible for:
- Creating and managing FAISS indices
- Adding vectors to indices
- Searching for similar vectors
- Persisting indices to disk
- Maintaining vector_id -> image_id mapping

Phase 2 Implementation:
- Two separate indices: text.index and image.index
- IndexFlatIP for cosine similarity (with L2-normalized vectors)
- Persistent storage on disk
"""
import json
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from app import config
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for FAISS
faiss = None


def _import_faiss():
    """Lazily import FAISS."""
    global faiss
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss


@dataclass
class SearchResult:
    """Result of a similarity search."""
    vector_id: int
    score: float  # Similarity score (higher = more similar)


class FAISSStore:
    """
    FAISS index manager for vector storage and retrieval.
    
    Features:
    - Two separate indices: text (caption embeddings) and image (pixel embeddings)
    - IndexFlatIP for exact cosine similarity search
    - Persistent storage to disk
    - Vector ID to image_id mapping
    """
    
    # Paths for index persistence
    TEXT_INDEX_FILENAME = "text.index"
    IMAGE_INDEX_FILENAME = "image.index"
    MAPPING_FILENAME = "vector_mapping.json"
    
    def __init__(self, dimension: int = 768):
        """
        Initialize FAISS store.
        
        Args:
            dimension: Embedding dimension (768 for CLIP ViT-L/14).
        """
        self._dimension = dimension
        self._text_index = None
        self._image_index = None
        self._is_loaded = False
        
        # Mapping: vector_id -> image_id (separate for text and image indices)
        self._text_id_to_image_id: Dict[int, str] = {}
        self._image_id_to_image_id: Dict[int, str] = {}
        
        # Reverse mapping for quick lookup
        self._image_id_to_text_vector: Dict[str, int] = {}
        self._image_id_to_image_vector: Dict[str, int] = {}
        
        logger.info(f"FAISS store initialized (dimension={dimension})")
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    @property
    def is_loaded(self) -> bool:
        """Check if indices are loaded."""
        return self._is_loaded
    
    def _get_index_paths(self) -> Tuple[Path, Path, Path]:
        """Get paths for index files."""
        text_path = config.FAISS_DIR / self.TEXT_INDEX_FILENAME
        image_path = config.FAISS_DIR / self.IMAGE_INDEX_FILENAME
        mapping_path = config.FAISS_DIR / self.MAPPING_FILENAME
        return text_path, image_path, mapping_path
    
    def create_indices(self) -> bool:
        """
        Create new empty FAISS indices.
        
        Uses IndexFlatIP for inner product (cosine similarity with normalized vectors).
        
        Returns:
            True if created successfully.
        """
        try:
            _import_faiss()
            
            # Create IndexFlatIP for inner product search
            # With L2-normalized vectors, IP = cosine similarity
            self._text_index = faiss.IndexFlatIP(self._dimension)
            self._image_index = faiss.IndexFlatIP(self._dimension)
            
            # Clear mappings
            self._text_id_to_image_id = {}
            self._image_id_to_image_id = {}
            self._image_id_to_text_vector = {}
            self._image_id_to_image_vector = {}
            
            self._is_loaded = True
            logger.info("Created new FAISS indices")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import FAISS: {e}")
            logger.error("Install with: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Failed to create indices: {e}")
            return False
    
    def load_indices(self) -> bool:
        """
        Load existing FAISS indices from disk.
        
        Returns:
            True if loaded successfully, False if not found or error.
        """
        try:
            _import_faiss()
            
            text_path, image_path, mapping_path = self._get_index_paths()
            
            # Check if indices exist
            if not text_path.exists() or not image_path.exists():
                logger.info("FAISS indices not found on disk, creating new ones")
                return self.create_indices()
            
            # Load indices
            self._text_index = faiss.read_index(str(text_path))
            self._image_index = faiss.read_index(str(image_path))
            
            # Load mappings
            if mapping_path.exists():
                with open(mapping_path, "r") as f:
                    mappings = json.load(f)
                
                # Convert string keys back to int
                self._text_id_to_image_id = {
                    int(k): v for k, v in mappings.get("text_to_image", {}).items()
                }
                self._image_id_to_image_id = {
                    int(k): v for k, v in mappings.get("image_to_image", {}).items()
                }
                
                # Build reverse mappings
                self._image_id_to_text_vector = {
                    v: k for k, v in self._text_id_to_image_id.items()
                }
                self._image_id_to_image_vector = {
                    v: k for k, v in self._image_id_to_image_id.items()
                }
            
            self._is_loaded = True
            logger.info(
                f"Loaded FAISS indices: {self._text_index.ntotal} text vectors, "
                f"{self._image_index.ntotal} image vectors"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            # Try creating new indices
            return self.create_indices()
    
    def save_indices(self) -> bool:
        """
        Save FAISS indices to disk.
        
        Returns:
            True if saved successfully.
        """
        if not self._is_loaded:
            logger.warning("Cannot save: indices not loaded")
            return False
        
        try:
            _import_faiss()
            
            text_path, image_path, mapping_path = self._get_index_paths()
            
            # Ensure directory exists
            config.FAISS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save indices
            faiss.write_index(self._text_index, str(text_path))
            faiss.write_index(self._image_index, str(image_path))
            
            # Save mappings
            mappings = {
                "text_to_image": {str(k): v for k, v in self._text_id_to_image_id.items()},
                "image_to_image": {str(k): v for k, v in self._image_id_to_image_id.items()}
            }
            with open(mapping_path, "w") as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(
                f"Saved FAISS indices: {self._text_index.ntotal} text, "
                f"{self._image_index.ntotal} image vectors"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save indices: {e}")
            return False
    
    def _ensure_loaded(self) -> bool:
        """Ensure indices are loaded, loading from disk if needed."""
        if not self._is_loaded:
            return self.load_indices()
        return True
    
    def add_text_vector(
        self, 
        vector: np.ndarray,
        image_id: str
    ) -> Optional[int]:
        """
        Add a text embedding vector to the text index.
        
        Args:
            vector: L2-normalized embedding vector (768,).
            image_id: Associated image ID (SHA256 hash).
            
        Returns:
            Vector ID in the index, or None if failed.
        """
        if not self._ensure_loaded():
            return None
        
        try:
            # Check if already exists
            if image_id in self._image_id_to_text_vector:
                logger.debug(f"Text vector for {image_id[:16]}... already exists")
                return self._image_id_to_text_vector[image_id]
            
            # Reshape vector for FAISS
            vec = vector.reshape(1, -1).astype(np.float32)
            
            # Get next vector ID (FAISS uses sequential IDs)
            vector_id = self._text_index.ntotal
            
            # Add to index
            self._text_index.add(vec)
            
            # Update mappings
            self._text_id_to_image_id[vector_id] = image_id
            self._image_id_to_text_vector[image_id] = vector_id
            
            return vector_id
            
        except Exception as e:
            logger.error(f"Failed to add text vector: {e}")
            return None
    
    def add_image_vector(
        self, 
        vector: np.ndarray,
        image_id: str
    ) -> Optional[int]:
        """
        Add an image embedding vector to the image index.
        
        Args:
            vector: L2-normalized embedding vector (768,).
            image_id: Associated image ID (SHA256 hash).
            
        Returns:
            Vector ID in the index, or None if failed.
        """
        if not self._ensure_loaded():
            return None
        
        try:
            # Check if already exists
            if image_id in self._image_id_to_image_vector:
                logger.debug(f"Image vector for {image_id[:16]}... already exists")
                return self._image_id_to_image_vector[image_id]
            
            # Reshape vector for FAISS
            vec = vector.reshape(1, -1).astype(np.float32)
            
            # Get next vector ID
            vector_id = self._image_index.ntotal
            
            # Add to index
            self._image_index.add(vec)
            
            # Update mappings
            self._image_id_to_image_id[vector_id] = image_id
            self._image_id_to_image_vector[image_id] = vector_id
            
            return vector_id
            
        except Exception as e:
            logger.error(f"Failed to add image vector: {e}")
            return None
    
    def search_text(
        self, 
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar text embeddings.
        
        Args:
            query_vector: L2-normalized query embedding vector.
            k: Number of results to return.
            
        Returns:
            List of SearchResult objects sorted by score descending.
        """
        if not self._ensure_loaded():
            return []
        
        if self._text_index.ntotal == 0:
            logger.warning("Text index is empty")
            return []
        
        try:
            # Reshape query
            query = query_vector.reshape(1, -1).astype(np.float32)
            
            # Limit k to available vectors
            k = min(k, self._text_index.ntotal)
            
            # Search
            scores, indices = self._text_index.search(query, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # FAISS returns -1 for missing results
                    results.append(SearchResult(vector_id=int(idx), score=float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def search_image(
        self, 
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search for similar image embeddings.
        
        Args:
            query_vector: L2-normalized query embedding vector.
            k: Number of results to return.
            
        Returns:
            List of SearchResult objects sorted by score descending.
        """
        if not self._ensure_loaded():
            return []
        
        if self._image_index.ntotal == 0:
            logger.warning("Image index is empty")
            return []
        
        try:
            # Reshape query
            query = query_vector.reshape(1, -1).astype(np.float32)
            
            # Limit k to available vectors
            k = min(k, self._image_index.ntotal)
            
            # Search
            scores, indices = self._image_index.search(query, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append(SearchResult(vector_id=int(idx), score=float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []
    
    def get_image_id_from_text_vector(self, vector_id: int) -> Optional[str]:
        """Get image_id from text vector ID."""
        return self._text_id_to_image_id.get(vector_id)
    
    def get_image_id_from_image_vector(self, vector_id: int) -> Optional[str]:
        """Get image_id from image vector ID."""
        return self._image_id_to_image_id.get(vector_id)
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining text and image similarity.
        
        Args:
            query_vector: L2-normalized query embedding vector.
            k: Number of final results to return.
            text_weight: Weight for text similarity (default 0.7).
            image_weight: Weight for image similarity (default 0.3).
            
        Returns:
            List of (image_id, combined_score) tuples, sorted by score descending.
        """
        # Search more than k to allow for combination
        search_k = k * 3
        
        # Search both indices
        text_results = self.search_text(query_vector, search_k)
        image_results = self.search_image(query_vector, search_k)
        
        # Combine scores by image_id
        scores: Dict[str, Dict[str, float]] = {}
        
        # Add text scores
        for result in text_results:
            image_id = self.get_image_id_from_text_vector(result.vector_id)
            if image_id:
                if image_id not in scores:
                    scores[image_id] = {"text": 0.0, "image": 0.0}
                scores[image_id]["text"] = result.score
        
        # Add image scores
        for result in image_results:
            image_id = self.get_image_id_from_image_vector(result.vector_id)
            if image_id:
                if image_id not in scores:
                    scores[image_id] = {"text": 0.0, "image": 0.0}
                scores[image_id]["image"] = result.score
        
        # Compute weighted combined scores
        combined = []
        for image_id, score_dict in scores.items():
            # If text score is missing but image exists, use only image score
            # If image score is missing but text exists, use only text score
            text_score = score_dict["text"]
            image_score = score_dict["image"]
            
            if text_score > 0 and image_score > 0:
                final_score = text_weight * text_score + image_weight * image_score
            elif text_score > 0:
                final_score = text_score  # Only text available
            else:
                final_score = image_score  # Only image available
            
            combined.append((image_id, final_score))
        
        # Sort by score descending
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:k]
    
    def get_text_vector_count(self) -> int:
        """Get number of vectors in text index."""
        if not self._is_loaded:
            return 0
        return self._text_index.ntotal if self._text_index else 0
    
    def get_image_vector_count(self) -> int:
        """Get number of vectors in image index."""
        if not self._is_loaded:
            return 0
        return self._image_index.ntotal if self._image_index else 0
    
    def has_vectors_for_image(self, image_id: str) -> Tuple[bool, bool]:
        """
        Check if vectors exist for an image.
        
        Args:
            image_id: Image ID to check.
            
        Returns:
            Tuple of (has_text_vector, has_image_vector).
        """
        has_text = image_id in self._image_id_to_text_vector
        has_image = image_id in self._image_id_to_image_vector
        return has_text, has_image


# Singleton instance
_faiss_store: Optional[FAISSStore] = None


def get_faiss_store() -> FAISSStore:
    """Get the singleton FAISS store instance."""
    global _faiss_store
    
    if _faiss_store is None:
        _faiss_store = FAISSStore()
    
    return _faiss_store
