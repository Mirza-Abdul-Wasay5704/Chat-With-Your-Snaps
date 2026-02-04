"""
Embedding service for the Memories Retrieval System.

Responsible for generating embeddings using CLIP ViT-L/14:
- Text embeddings from captions/queries
- Image embeddings from pixel data

Phase 2 Implementation:
- Output dimension: 768
- L2-normalized vectors (for cosine similarity via inner product)
- CPU-friendly inference
"""
import io
from typing import Optional, List, Union
import numpy as np
from dataclasses import dataclass
from PIL import Image

from app import config
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for ML libraries
torch = None
CLIPModel = None
CLIPProcessor = None


def _import_ml_libraries():
    """Lazily import ML libraries to avoid startup overhead."""
    global torch, CLIPModel, CLIPProcessor
    
    if torch is None:
        import torch as _torch
        from transformers import CLIPModel as _CLIPModel
        from transformers import CLIPProcessor as _CLIPProcessor
        
        torch = _torch
        CLIPModel = _CLIPModel
        CLIPProcessor = _CLIPProcessor


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector for cosine similarity.
    
    Args:
        vector: Input vector of any shape.
        
    Returns:
        Normalized vector with unit L2 norm.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    success: bool
    embedding: Optional[np.ndarray]  # Shape: (768,), L2-normalized
    error: Optional[str] = None


class EmbedderService:
    """
    Service for generating CLIP embeddings.
    
    Features:
    - CLIP ViT-L/14 model (768-dim embeddings)
    - L2-normalized output (for FAISS IndexFlatIP)
    - CPU-only inference (GPU optional)
    - Batch processing support
    """
    
    # CLIP ViT-L/14 embedding dimension
    EMBEDDING_DIM = 768
    
    def __init__(self):
        """Initialize the embedder service."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._device = None
        logger.info("Embedder service initialized (model not loaded)")
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.EMBEDDING_DIM
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def load_model(self) -> bool:
        """
        Load the CLIP model.
        
        Returns:
            True if loaded successfully.
        """
        if self._is_loaded:
            logger.info("Embedder model already loaded")
            return True
        
        try:
            _import_ml_libraries()
            
            # Determine device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading CLIP model ({config.CLIP_MODEL_NAME}) on {self._device}...")
            
            # Load processor
            self._processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
            
            # Load model
            self._model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
            self._model = self._model.to(self._device).eval()
            
            self._is_loaded = True
            logger.info(f"CLIP model loaded successfully on {self._device}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ML libraries: {e}")
            logger.error("Install with: pip install torch transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return False
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate L2-normalized embedding for text (caption or query).
        
        Args:
            text: Text to embed.
            
        Returns:
            EmbeddingResult with normalized embedding vector (768,).
        """
        if not self._is_loaded:
            if not self.load_model():
                return EmbeddingResult(
                    success=False,
                    embedding=None,
                    error="Failed to load embedding model"
                )
        
        try:
            # Prepare text input
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP max context length
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = text_features.cpu().numpy()[0]
            embedding = normalize_vector(embedding.astype(np.float32))
            
            return EmbeddingResult(
                success=True,
                embedding=embedding
            )
            
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return EmbeddingResult(
                success=False,
                embedding=None,
                error=str(e)
            )
    
    def embed_texts_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of EmbeddingResult objects.
        """
        if not self._is_loaded:
            if not self.load_model():
                return [
                    EmbeddingResult(success=False, embedding=None, error="Model not loaded")
                    for _ in texts
                ]
        
        results = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self._model.get_text_features(**inputs)
                
                embeddings = text_features.cpu().numpy().astype(np.float32)
                
                for emb in embeddings:
                    normalized = normalize_vector(emb)
                    results.append(EmbeddingResult(success=True, embedding=normalized))
                    
            except Exception as e:
                logger.error(f"Batch text embedding failed: {e}")
                for _ in batch:
                    results.append(EmbeddingResult(success=False, embedding=None, error=str(e)))
        
        return results
    
    def _open_image(self, image_input: Union[bytes, str]) -> Image.Image:
        """
        Open image from bytes or file path.
        
        Args:
            image_input: Either raw bytes or file path.
            
        Returns:
            PIL Image in RGB mode.
        """
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            return Image.open(image_input).convert("RGB")
    
    def embed_image(self, image_input: Union[bytes, str]) -> EmbeddingResult:
        """
        Generate L2-normalized embedding for an image.
        
        Args:
            image_input: Raw JPEG bytes or file path.
            
        Returns:
            EmbeddingResult with normalized embedding vector (768,).
        """
        if not self._is_loaded:
            if not self.load_model():
                return EmbeddingResult(
                    success=False,
                    embedding=None,
                    error="Failed to load embedding model"
                )
        
        try:
            # Open image
            image = self._open_image(image_input)
            
            # Prepare image input
            inputs = self._processor(
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items() if k == "pixel_values"}
            
            # Generate embedding
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = image_features.cpu().numpy()[0]
            embedding = normalize_vector(embedding.astype(np.float32))
            
            image.close()
            
            return EmbeddingResult(
                success=True,
                embedding=embedding
            )
            
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return EmbeddingResult(
                success=False,
                embedding=None,
                error=str(e)
            )
    
    def embed_images_batch(
        self, 
        images: List[Union[bytes, str]]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple images.
        
        Processes images in batches for efficiency.
        
        Args:
            images: List of image bytes or file paths.
            
        Returns:
            List of EmbeddingResult objects.
        """
        if not self._is_loaded:
            if not self.load_model():
                return [
                    EmbeddingResult(success=False, embedding=None, error="Model not loaded")
                    for _ in images
                ]
        
        results = []
        batch_size = 8  # Smaller batch for images (memory)
        
        for i in range(0, len(images), batch_size):
            batch_inputs = images[i:i + batch_size]
            
            try:
                # Open all images in batch
                pil_images = [self._open_image(img) for img in batch_inputs]
                
                inputs = self._processor(
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
                pixel_values = inputs["pixel_values"].to(self._device)
                
                with torch.no_grad():
                    image_features = self._model.get_image_features(pixel_values=pixel_values)
                
                embeddings = image_features.cpu().numpy().astype(np.float32)
                
                for emb in embeddings:
                    normalized = normalize_vector(emb)
                    results.append(EmbeddingResult(success=True, embedding=normalized))
                
                # Cleanup
                for img in pil_images:
                    img.close()
                    
            except Exception as e:
                logger.error(f"Batch image embedding failed: {e}")
                for _ in batch_inputs:
                    results.append(EmbeddingResult(success=False, embedding=None, error=str(e)))
        
        return results
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        self._is_loaded = False
        
        # Clear CUDA cache if available
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Embedder model unloaded")


# Singleton instance
_embedder_service: Optional[EmbedderService] = None


def get_embedder_service() -> EmbedderService:
    """Get the singleton embedder service instance."""
    global _embedder_service
    
    if _embedder_service is None:
        _embedder_service = EmbedderService()
    
    return _embedder_service
