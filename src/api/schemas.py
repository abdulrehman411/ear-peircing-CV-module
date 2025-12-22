"""
Pydantic schemas for API requests and responses.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from src.models.ear import Point, BoundingBox, EarDimensions, Landmark
from src.models.piercing import Piercing
from src.models.validation import Offset, Adjustment, ValidationResult


# Request Schemas
class EarDetectionRequest(BaseModel):
    """Request for ear detection."""
    image: str = Field(..., description="Base64 encoded image")


class PiercingDetectionRequest(BaseModel):
    """Request for piercing detection."""
    image: str = Field(..., description="Base64 encoded image")
    ear_landmarks: List[dict] = Field(..., description="Ear landmarks from detection")
    bounding_box: dict = Field(..., description="Ear bounding box")


class PointMarkRequest(BaseModel):
    """Request for marking a point."""
    image: str = Field(..., description="Base64 encoded image")
    point: dict = Field(..., description="Point coordinates {x, y} (normalized 0-1)")
    ear_landmarks: List[dict] = Field(..., description="Ear landmarks")
    bounding_box: dict = Field(..., description="Ear bounding box")


class ValidationRequest(BaseModel):
    """Request for point validation."""
    original_image: str = Field(..., description="Original image (base64)")
    rescan_image: str = Field(..., description="Re-scanned image with mark (base64)")
    digital_point: dict = Field(..., description="Digital point coordinates")
    ear_landmarks: List[dict] = Field(..., description="Ear landmarks")
    bounding_box: dict = Field(..., description="Ear bounding box")


class FeedbackRequest(BaseModel):
    """Request for real-time feedback."""
    rescan_image: str = Field(..., description="Re-scanned image (base64)")
    expected_point: dict = Field(..., description="Expected point coordinates")
    ear_landmarks: List[dict] = Field(..., description="Ear landmarks")
    bounding_box: dict = Field(..., description="Ear bounding box")


class SymmetryMapRequest(BaseModel):
    """Request for symmetry mapping."""
    ear1_image: str = Field(..., description="First ear image (base64)")
    ear2_image: str = Field(..., description="Second ear image (base64)")
    ear1_point: dict = Field(..., description="Point on first ear")
    ear1_landmarks: List[dict] = Field(..., description="First ear landmarks")
    ear2_landmarks: List[dict] = Field(..., description="Second ear landmarks")
    ear1_bounding_box: dict = Field(..., description="First ear bounding box")
    ear2_bounding_box: dict = Field(..., description="Second ear bounding box")


class RescanValidateRequest(BaseModel):
    """Request for re-scan validation."""
    rescan_image: str = Field(..., description="Re-scanned image (base64)")
    original_point: dict = Field(..., description="Original digital point")
    validation_history: List[dict] = Field(default=[], description="Previous validation results")
    ear_landmarks: List[dict] = Field(..., description="Ear landmarks")
    bounding_box: dict = Field(..., description="Ear bounding box")


# Response Schemas
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"


class EarDetectionResponse(BaseModel):
    """Response for ear detection."""
    success: bool
    ear_detected: bool
    confidence: float
    dimensions: Optional[dict] = None
    landmarks: List[dict] = []
    bounding_box: Optional[dict] = None
    ear_side: Optional[str] = None


class PiercingDetectionResponse(BaseModel):
    """Response for piercing detection."""
    success: bool
    piercings: List[dict] = []
    total_count: int = 0


class PointMarkResponse(BaseModel):
    """Response for point marking."""
    success: bool
    point: dict
    landmark_references: dict = {}
    normalized: bool = True


class ValidationResponse(BaseModel):
    """Response for point validation."""
    success: bool
    valid: bool
    offset: dict
    feedback: str
    adjustment: Optional[dict] = None
    confidence: float = 1.0


class FeedbackResponse(BaseModel):
    """Response for feedback."""
    success: bool
    status: str
    message: str
    offset: dict
    adjustment: Optional[dict] = None


class SymmetryMapResponse(BaseModel):
    """Response for symmetry mapping."""
    success: bool
    mapped_point: dict
    scale_factor: float
    scale_x: float
    scale_y: float
    ear2_dimensions: dict


class RescanValidateResponse(BaseModel):
    """Response for re-scan validation."""
    success: bool
    validated: bool
    iterations: int
    final_offset: dict
    best_result: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    message: str
    details: Optional[dict] = None

