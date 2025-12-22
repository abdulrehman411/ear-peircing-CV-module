"""
Coordinate transformation utilities.
"""
import numpy as np
from typing import Tuple, List
from src.models.ear import Point, BoundingBox


def pixel_to_normalized(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Convert pixel coordinates to normalized coordinates (0-1).
    
    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        width: Image width
        height: Image height
        
    Returns:
        Normalized coordinates (x, y)
    """
    norm_x = x / width if width > 0 else 0.0
    norm_y = y / height if height > 0 else 0.0
    return (norm_x, norm_y)


def normalized_to_pixel(norm_x: float, norm_y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Convert normalized coordinates to pixel coordinates.
    
    Args:
        norm_x: Normalized X coordinate (0-1)
        norm_y: Normalized Y coordinate (0-1)
        width: Image width
        height: Image height
        
    Returns:
        Pixel coordinates (x, y)
    """
    x = norm_x * width
    y = norm_y * height
    return (x, y)


def to_ear_relative(point: Point, bbox: BoundingBox) -> Point:
    """
    Convert point to ear-relative coordinates.
    
    Args:
        point: Point in image coordinates
        bbox: Ear bounding box
        
    Returns:
        Point in ear-relative coordinates
    """
    # Normalize relative to bounding box
    ear_width = bbox.width
    ear_height = bbox.height
    
    if ear_width > 0 and ear_height > 0:
        rel_x = (point.x - bbox.x_min) / ear_width
        rel_y = (point.y - bbox.y_min) / ear_height
    else:
        rel_x = point.x
        rel_y = point.y
    
    return Point(x=rel_x, y=rel_y)


def from_ear_relative(point: Point, bbox: BoundingBox) -> Point:
    """
    Convert ear-relative point to image coordinates.
    
    Args:
        point: Point in ear-relative coordinates
        bbox: Ear bounding box
        
    Returns:
        Point in image coordinates
    """
    x = bbox.x_min + point.x * bbox.width
    y = bbox.y_min + point.y * bbox.height
    return Point(x=x, y=y)


def calculate_distance(point1: Point, point2: Point) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point
        point2: Second point
        
    Returns:
        Distance
    """
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return np.sqrt(dx * dx + dy * dy)


def mirror_point(point: Point, width: float = 1.0) -> Point:
    """
    Mirror point across vertical axis (for symmetry mapping).
    
    Args:
        point: Point to mirror (normalized coordinates)
        width: Image width (default 1.0 for normalized)
        
    Returns:
        Mirrored point
    """
    mirrored_x = width - point.x
    return Point(x=mirrored_x, y=point.y)


def apply_scaling(point: Point, scale_x: float, scale_y: float) -> Point:
    """
    Apply scaling to point coordinates.
    
    Args:
        point: Point to scale
        scale_x: X-axis scale factor
        scale_y: Y-axis scale factor
        
    Returns:
        Scaled point
    """
    return Point(x=point.x * scale_x, y=point.y * scale_y)


def transform_for_symmetry(
    point: Point,
    ear1_dimensions: Tuple[float, float],
    ear2_dimensions: Tuple[float, float],
    image_width: float = 1.0
) -> Point:
    """
    Transform point from first ear to second ear using symmetry mapping.
    
    Args:
        point: Point on first ear (normalized)
        ear1_dimensions: (length, width) of first ear
        ear2_dimensions: (length, width) of second ear
        image_width: Image width (default 1.0 for normalized)
        
    Returns:
        Transformed point for second ear
    """
    # Calculate scale factors
    scale_x = ear2_dimensions[1] / ear1_dimensions[1] if ear1_dimensions[1] > 0 else 1.0
    scale_y = ear2_dimensions[0] / ear1_dimensions[0] if ear1_dimensions[0] > 0 else 1.0
    
    # Mirror the point (flip x-axis)
    mirrored = mirror_point(point, image_width)
    
    # Apply scaling
    scaled = apply_scaling(mirrored, scale_x, scale_y)
    
    return scaled


def normalize_landmarks(landmarks: List[Point], width: int, height: int) -> List[Point]:
    """
    Normalize landmark coordinates to 0-1 range.
    
    Args:
        landmarks: List of landmark points in pixel coordinates
        width: Image width
        height: Image height
        
    Returns:
        List of normalized landmark points
    """
    normalized = []
    for landmark in landmarks:
        norm_x, norm_y = pixel_to_normalized(landmark.x, landmark.y, width, height)
        normalized.append(Point(x=norm_x, y=norm_y))
    return normalized

