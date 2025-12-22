"""
Piercing detection data models.
"""
from typing import Optional
from pydantic import BaseModel
from .ear import Point


class Piercing(BaseModel):
    """Detected piercing information."""
    point: Point
    type: Optional[str] = None  # "stud", "hoop", "unknown"
    confidence: float
    size: Optional[float] = None  # Approximate size in normalized units


class PiercingDetectionResult(BaseModel):
    """Piercing detection result."""
    piercings: list[Piercing] = []
    total_count: int = 0

