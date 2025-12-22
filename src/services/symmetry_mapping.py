"""
Symmetry mapping service for replicating piercing points.
"""
from src.models.ear import EarDimensions, Point
from src.models.validation import PointMark
from src.utils.coordinate_transform import transform_for_symmetry


class SymmetryMappingService:
    """Service for mapping piercing points between ears."""
    
    def map_point_to_second_ear(
        self,
        ear1_point: Point,
        ear1_dimensions: EarDimensions,
        ear2_dimensions: EarDimensions,
        image_width: float = 1.0
    ) -> Point:
        """
        Map piercing point from first ear to second ear using symmetry.
        
        Args:
            ear1_point: Point on first ear (normalized)
            ear1_dimensions: Dimensions of first ear
            ear2_dimensions: Dimensions of second ear
            image_width: Image width (default 1.0 for normalized)
            
        Returns:
            Mapped point for second ear
        """
        # Transform point using symmetry
        mapped_point = transform_for_symmetry(
            ear1_point,
            (ear1_dimensions.length, ear1_dimensions.width),
            (ear2_dimensions.length, ear2_dimensions.width),
            image_width
        )
        
        return mapped_point
    
    def calculate_scale_factor(
        self,
        ear1_dimensions: EarDimensions,
        ear2_dimensions: EarDimensions
    ) -> dict:
        """
        Calculate scale factors between two ears.
        
        Args:
            ear1_dimensions: Dimensions of first ear
            ear2_dimensions: Dimensions of second ear
            
        Returns:
            Dictionary with scale factors
        """
        scale_x = ear2_dimensions.width / ear1_dimensions.width if ear1_dimensions.width > 0 else 1.0
        scale_y = ear2_dimensions.length / ear1_dimensions.length if ear1_dimensions.length > 0 else 1.0
        
        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "scale_factor": (scale_x + scale_y) / 2.0
        }

