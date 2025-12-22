"""
Ear detection data models.
"""
from typing import List, Optional
from pydantic import BaseModel


class Point(BaseModel):
    """2D point coordinates."""
    x: float
    y: float


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class Landmark(BaseModel):
    """Ear landmark point."""
    x: float
    y: float
    z: Optional[float] = None
    visibility: Optional[float] = None
    index: Optional[int] = None


class EarDimensions(BaseModel):
    """Ear dimension measurements."""
    length: float  # Normalized (0-1)
    width: float   # Normalized (0-1)
    length_pixels: Optional[float] = None
    width_pixels: Optional[float] = None


class EarDetectionResult(BaseModel):
    """Ear detection result."""
    ear_detected: bool
    confidence: float
    dimensions: Optional[EarDimensions] = None
    landmarks: List[Landmark] = []
    bounding_box: Optional[BoundingBox] = None
    ear_side: Optional[str] = None  # "left" or "right"

