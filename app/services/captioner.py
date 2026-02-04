"""
Captioning service for the Memories Retrieval System.

Responsible for generating image captions using Florence-2.

Phase 2 Implementation:
- Load Florence-2 model once (CPU-friendly)
- Generate detailed captions for images
- Store captions in SQLite via indexer
"""
import io
from typing import Optional, List, Union
from dataclasses import dataclass
from PIL import Image

from app import config
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for ML libraries (only when model is loaded)
torch = None
AutoProcessor = None
AutoModelForCausalLM = None


def _import_ml_libraries():
    """Lazily import ML libraries to avoid startup overhead."""
    global torch, AutoProcessor, AutoModelForCausalLM
    
    if torch is None:
        import torch as _torch
        from transformers import AutoProcessor as _AutoProcessor
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        
        torch = _torch
        AutoProcessor = _AutoProcessor
        AutoModelForCausalLM = _AutoModelForCausalLM


@dataclass
class CaptionResult:
    """Result of caption generation."""
    image_id: str
    caption: Optional[str]
    success: bool
    error: Optional[str] = None


class CaptionerService:
    """
    Service for generating image captions using Florence-2.
    
    Features:
    - CPU-only inference (GPU optional)
    - Deterministic output
    - Batch-friendly processing
    """
    
    def __init__(self):
        """Initialize the captioner service."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._device = None
        self._dtype = None
        logger.info("Captioner service initialized (model not loaded)")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def load_model(self) -> bool:
        """
        Load the Florence-2 model.
        
        Uses CPU by default, GPU if available.
        
        Returns:
            True if loaded successfully.
        """
        if self._is_loaded:
            logger.info("Captioner model already loaded")
            return True
        
        try:
            _import_ml_libraries()
            
            # Determine device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._dtype = torch.float16 if self._device == "cuda" else torch.float32
            
            logger.info(f"Loading Florence-2 model on {self._device}...")
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                config.FLORENCE_MODEL_NAME,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            self._model = AutoModelForCausalLM.from_pretrained(
                config.FLORENCE_MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=self._dtype,
                attn_implementation="eager"  # Prevents SDPA errors
            ).to(self._device).eval()
            
            self._is_loaded = True
            logger.info(f"Florence-2 model loaded successfully on {self._device}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ML libraries: {e}")
            logger.error("Install with: pip install torch transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            return False
    
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
    
    def generate_caption(
        self, 
        image_input: Union[bytes, str],
        image_id: str
    ) -> CaptionResult:
        """
        Generate a detailed caption for an image.
        
        Args:
            image_input: Raw JPEG bytes or file path.
            image_id: Unique identifier for the image.
            
        Returns:
            CaptionResult with generated caption.
        """
        if not self._is_loaded:
            if not self.load_model():
                return CaptionResult(
                    image_id=image_id,
                    caption=None,
                    success=False,
                    error="Failed to load captioning model"
                )
        
        try:
            # Open and prepare image
            image = self._open_image(image_input)
            
            # Prepare inputs with detailed caption prompt
            inputs = self._processor(
                text="<DETAILED_CAPTION>",
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            input_ids = inputs["input_ids"].to(self._device)
            pixel_values = inputs["pixel_values"].to(self._device, dtype=self._dtype)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False,  # Deterministic
                    use_cache=False
                )
            
            # Decode caption
            caption = self._processor.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0]
            
            # Clean up the caption (remove prompt artifacts)
            caption = caption.strip()
            if caption.startswith("<DETAILED_CAPTION>"):
                caption = caption[len("<DETAILED_CAPTION>"):].strip()
            
            image.close()
            
            logger.debug(f"Generated caption for {image_id[:16]}...: {caption[:50]}...")
            
            return CaptionResult(
                image_id=image_id,
                caption=caption,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Caption generation failed for {image_id[:16]}...: {e}")
            return CaptionResult(
                image_id=image_id,
                caption=None,
                success=False,
                error=str(e)
            )
    
    def generate_captions_batch(
        self,
        images: List[tuple[Union[bytes, str], str]]  # List of (image_input, image_id)
    ) -> List[CaptionResult]:
        """
        Generate captions for multiple images.
        
        Processes images one at a time to manage memory.
        
        Args:
            images: List of (image_input, image_id) tuples.
            
        Returns:
            List of CaptionResult objects.
        """
        results = []
        
        for idx, (image_input, image_id) in enumerate(images):
            result = self.generate_caption(image_input, image_id)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Captioned {idx + 1}/{len(images)} images")
        
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
        
        logger.info("Captioner model unloaded")


# Singleton instance
_captioner_service: Optional[CaptionerService] = None


def get_captioner_service() -> CaptionerService:
    """Get the singleton captioner service instance."""
    global _captioner_service
    
    if _captioner_service is None:
        _captioner_service = CaptionerService()
    
    return _captioner_service
