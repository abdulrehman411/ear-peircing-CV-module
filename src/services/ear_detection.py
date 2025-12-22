"""
Ear detection service using MediaPipe.
"""
import numpy as np
from typing import Optional
from src.config import get_settings
from src.models.ear import EarDetectionResult, EarDimensions, BoundingBox, Landmark
from src.utils.mediapipe_wrapper import MediaPipeWrapper
from src.utils.coordinate_transform import pixel_to_normalized, normalize_landmarks


class EarDetectionService:
    """Service for detecting ears in images."""
    
    def __init__(self):
        self.settings = get_settings()
        self.mediapipe = MediaPipeWrapper()
    
    def detect_ear(self, image: np.ndarray) -> EarDetectionResult:
        """
        Detect ear in image and extract landmarks.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            EarDetectionResult with detection information
        """
        height, width = image.shape[:2]
        
        # Detect ear side
        ear_side = self.mediapipe.detect_ear_side(image)
        
        # Extract ear landmarks
        ear_landmarks = self.mediapipe.extract_ear_landmarks(image, ear_side or "left")
        
        if ear_landmarks is None or len(ear_landmarks) == 0:
            return EarDetectionResult(
                ear_detected=False,
                confidence=0.0
            )
        
        # Calculate bounding box
        bbox_coords = self.mediapipe.get_ear_bounding_box(ear_landmarks)
        
        if bbox_coords is None:
            return EarDetectionResult(
                ear_detected=False,
                confidence=0.0
            )
        
        x_min, y_min, x_max, y_max = bbox_coords
        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=x_max - x_min,
            height=y_max - y_min
        )
        
        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(ear_landmarks, width, height)
        
        # Calculate confidence based on landmark visibility
        avg_visibility = sum(l.visibility or 0.5 for l in ear_landmarks) / len(ear_landmarks)
        confidence = min(avg_visibility, 1.0)
        
        return EarDetectionResult(
            ear_detected=True,
            confidence=confidence,
            landmarks=normalized_landmarks,
            bounding_box=bbox,
            ear_side=ear_side
        )

