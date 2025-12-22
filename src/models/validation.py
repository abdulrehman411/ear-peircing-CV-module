"""
Validation data models.
"""
from typing import Optional, List
from pydantic import BaseModel
from .ear import Point


class Offset(BaseModel):
    """Offset between two points."""
    x: float
    y: float
    distance: float  # Euclidean distance


class Adjustment(BaseModel):
    """Adjustment feedback."""
    direction: str  # "left", "right", "up", "down", "correct", or combinations
    magnitude: float
    units: str = "normalized"


class ValidationResult(BaseModel):
    """Point validation result."""
    valid: bool
    offset: Offset
    feedback: str  # "correct", "adjust_left", "adjust_right", etc.
    adjustment: Optional[Adjustment] = None
    confidence: float = 1.0


class PointMark(BaseModel):
    """Digital point mark."""
    point: Point
    landmark_references: dict = {}  # Distances to key landmarks
    normalized: bool = True
    timestamp: Optional[float] = None


class ValidationHistory(BaseModel):
    """Validation history for re-scan iterations."""
    iterations: List[ValidationResult] = []
    total_iterations: int = 0
    best_result: Optional[ValidationResult] = None

