"""
Image preprocessing pipeline.
"""
import numpy as np
import gc
from typing import Optional
from src.config import get_settings
from src.utils.image_utils import (
    base64_to_image,
    resize_image,
    normalize_lighting,
    reduce_noise,
    enhance_contrast,
    crop_region
)


class ImageProcessor:
    """Image preprocessing service."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def process_image(self, base64_string: str, preprocess: bool = True) -> np.ndarray:
        """
        Process image from base64 string with optimized memory management.
        
        Args:
            base64_string: Base64 encoded image
            preprocess: Whether to apply preprocessing
            
        Returns:
            Processed image as numpy array
        """
        # Convert base64 to numpy array
        image = base64_to_image(base64_string)
        
        # Validate image size early to avoid unnecessary processing
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Image has invalid dimensions")
        
        if not preprocess:
            return image
        
        try:
            # Resize if needed (do this first to reduce memory usage)
            if self.settings.max_image_size > 0:
                original_size = image.nbytes
                image = resize_image(image, self.settings.max_image_size)
                # Force garbage collection if significant size reduction
                if original_size > image.nbytes * 2:
                    gc.collect()
            
            # Normalize lighting if enabled
            if self.settings.lighting_correction_enabled:
                image = normalize_lighting(image)
            
            # Reduce noise (in-place operations where possible)
            image = reduce_noise(image)
            
            return image
        except Exception as e:
            # Clean up on error
            del image
            gc.collect()
            raise
    
    def extract_ear_region(self, image: np.ndarray, bbox: tuple, padding: int = 20) -> np.ndarray:
        """
        Extract ear region from image.
        
        Args:
            image: Full image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            padding: Padding around bounding box
            
        Returns:
            Cropped ear region
        """
        return crop_region(image, bbox, padding)
    
    def prepare_for_mark_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for mark detection (enhance contrast).
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        return enhance_contrast(image, alpha=1.5, beta=10)
    
    def prepare_for_piercing_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for piercing detection.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Enhance contrast for better detection
        enhanced = enhance_contrast(image, alpha=1.3, beta=5)
        return enhanced

