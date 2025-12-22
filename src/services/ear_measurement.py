"""
Ear dimension measurement service.
"""
import numpy as np
from typing import Tuple
from src.models.ear import EarDimensions, EarDetectionResult, Landmark
from src.utils.coordinate_transform import calculate_distance


class EarMeasurementService:
    """Service for measuring ear dimensions."""
    
    def measure_dimensions(self, detection_result: EarDetectionResult, image_width: int, image_height: int) -> EarDimensions:
        """
        Measure ear length and width from detection result.
        
        Args:
            detection_result: Ear detection result with landmarks
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            EarDimensions with normalized and pixel measurements
        """
        if not detection_result.ear_detected or not detection_result.landmarks:
            return EarDimensions(length=0.0, width=0.0)
        
        landmarks = detection_result.landmarks
        
        # Convert normalized landmarks back to pixels for calculation
        pixel_landmarks = []
        for landmark in landmarks:
            pixel_x = landmark.x * image_width
            pixel_y = landmark.y * image_height
            pixel_landmarks.append((pixel_x, pixel_y))
        
        # Calculate length (vertical distance)
        y_coords = [y for _, y in pixel_landmarks]
        length_pixels = max(y_coords) - min(y_coords)
        
        # Calculate width (horizontal distance)
        x_coords = [x for x, _ in pixel_landmarks]
        width_pixels = max(x_coords) - min(x_coords)
        
        # Normalize to image dimensions
        length = length_pixels / image_height if image_height > 0 else 0.0
        width = width_pixels / image_width if image_width > 0 else 0.0
        
        return EarDimensions(
            length=length,
            width=width,
            length_pixels=length_pixels,
            width_pixels=width_pixels
        )
    
    def calculate_scale_factor(self, ear1_dimensions: EarDimensions, ear2_dimensions: EarDimensions) -> Tuple[float, float]:
        """
        Calculate scale factors between two ears.
        
        Args:
            ear1_dimensions: Dimensions of first ear
            ear2_dimensions: Dimensions of second ear
            
        Returns:
            Tuple of (scale_x, scale_y)
        """
        scale_x = ear2_dimensions.width / ear1_dimensions.width if ear1_dimensions.width > 0 else 1.0
        scale_y = ear2_dimensions.length / ear1_dimensions.length if ear1_dimensions.length > 0 else 1.0
        
        return (scale_x, scale_y)

