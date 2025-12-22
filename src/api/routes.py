"""
API routes for the CV module.
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.api.schemas import (
    EarDetectionRequest,
    EarDetectionResponse,
    PiercingDetectionRequest,
    PiercingDetectionResponse,
    PointMarkRequest,
    PointMarkResponse,
    ValidationRequest,
    ValidationResponse,
    FeedbackRequest,
    FeedbackResponse,
    SymmetryMapRequest,
    SymmetryMapResponse,
    RescanValidateRequest,
    RescanValidateResponse,
    HealthResponse,
    ErrorResponse
)
from src.services.image_processor import ImageProcessor
from src.services.ear_detection import EarDetectionService
from src.services.ear_measurement import EarMeasurementService
from src.services.piercing_detection import PiercingDetectionService
from src.services.point_validation import PointValidationService
from src.services.symmetry_mapping import SymmetryMappingService
from src.models.ear import Point, Landmark, BoundingBox, EarDimensions
from src.models.validation import ValidationResult

router = APIRouter()

# Initialize services
image_processor = ImageProcessor()
ear_detection = EarDetectionService()
ear_measurement = EarMeasurementService()
piercing_detection = PiercingDetectionService()
point_validation = PointValidationService()
symmetry_mapping = SymmetryMappingService()


def _dict_to_point(data: dict) -> Point:
    """Convert dict to Point."""
    return Point(x=data["x"], y=data["y"])


def _dict_to_landmark(data: dict) -> Landmark:
    """Convert dict to Landmark."""
    return Landmark(
        x=data.get("x", 0),
        y=data.get("y", 0),
        z=data.get("z"),
        visibility=data.get("visibility"),
        index=data.get("index")
    )


def _dict_to_bbox(data: dict) -> BoundingBox:
    """Convert dict to BoundingBox."""
    return BoundingBox(
        x_min=data["x_min"],
        y_min=data["y_min"],
        x_max=data["x_max"],
        y_max=data["y_max"],
        width=data.get("width", data["x_max"] - data["x_min"]),
        height=data.get("height", data["y_max"] - data["y_min"])
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.post("/detect-ear", response_model=EarDetectionResponse)
async def detect_ear(request: EarDetectionRequest):
    """Detect ear and measure dimensions."""
    try:
        # Process image
        image = image_processor.process_image(request.image)
        height, width = image.shape[:2]
        
        # Detect ear
        result = ear_detection.detect_ear(image)
        
        if not result.ear_detected:
            return EarDetectionResponse(
                success=False,
                ear_detected=False,
                confidence=0.0
            )
        
        # Measure dimensions
        dimensions = ear_measurement.measure_dimensions(result, width, height)
        
        # Convert to response format
        return EarDetectionResponse(
            success=True,
            ear_detected=True,
            confidence=result.confidence,
            dimensions={
                "length": dimensions.length,
                "width": dimensions.width,
                "length_pixels": dimensions.length_pixels,
                "width_pixels": dimensions.width_pixels
            },
            landmarks=[{"x": l.x, "y": l.y, "z": l.z, "visibility": l.visibility, "index": l.index} for l in result.landmarks],
            bounding_box={
                "x_min": result.bounding_box.x_min,
                "y_min": result.bounding_box.y_min,
                "x_max": result.bounding_box.x_max,
                "y_max": result.bounding_box.y_max,
                "width": result.bounding_box.width,
                "height": result.bounding_box.height
            } if result.bounding_box else None,
            ear_side=result.ear_side
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ear detection failed: {str(e)}")


@router.post("/detect-piercings", response_model=PiercingDetectionResponse)
async def detect_piercings(request: PiercingDetectionRequest):
    """Detect existing piercings."""
    try:
        # Process image
        image = image_processor.process_image(request.image)
        height, width = image.shape[:2]
        
        # Convert landmarks and bbox
        landmarks = [_dict_to_landmark(l) for l in request.ear_landmarks]
        bbox = _dict_to_bbox(request.bounding_box)
        
        # Detect piercings
        result = piercing_detection.detect_piercings(image, landmarks, bbox, width, height)
        
        return PiercingDetectionResponse(
            success=True,
            piercings=[{
                "point": {"x": p.point.x, "y": p.point.y},
                "type": p.type,
                "confidence": p.confidence,
                "size": p.size
            } for p in result.piercings],
            total_count=result.total_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Piercing detection failed: {str(e)}")


@router.post("/mark-point", response_model=PointMarkResponse)
async def mark_point(request: PointMarkRequest):
    """Mark a digital piercing point."""
    try:
        # Process image
        image = image_processor.process_image(request.image)
        height, width = image.shape[:2]
        
        # Convert data
        point = _dict_to_point(request.point)
        landmarks = [_dict_to_landmark(l) for l in request.ear_landmarks]
        bbox = _dict_to_bbox(request.bounding_box)
        
        # Mark point
        mark = point_validation.mark_point(point, landmarks, bbox, width, height)
        
        return PointMarkResponse(
            success=True,
            point={"x": mark.point.x, "y": mark.point.y},
            landmark_references=mark.landmark_references,
            normalized=mark.normalized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point marking failed: {str(e)}")


@router.post("/validate-point", response_model=ValidationResponse)
async def validate_point(request: ValidationRequest):
    """Validate physical point against digital point."""
    try:
        # Process images
        rescan_image = image_processor.process_image(request.rescan_image)
        height, width = rescan_image.shape[:2]
        
        # Convert data
        digital_point = _dict_to_point(request.digital_point)
        bbox = _dict_to_bbox(request.bounding_box)
        
        # Detect physical mark
        physical_point = point_validation.detect_physical_mark(
            rescan_image, digital_point, bbox, width, height
        )
        
        # Validate
        result = point_validation.validate_point(digital_point, physical_point, width, height)
        
        return ValidationResponse(
            success=True,
            valid=result.valid,
            offset={
                "x": result.offset.x,
                "y": result.offset.y,
                "distance": result.offset.distance
            },
            feedback=result.feedback,
            adjustment={
                "direction": result.adjustment.direction,
                "magnitude": result.adjustment.magnitude,
                "units": result.adjustment.units
            } if result.adjustment else None,
            confidence=result.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point validation failed: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def get_feedback(request: FeedbackRequest):
    """Get real-time feedback for point adjustment."""
    try:
        # Process image
        image = image_processor.process_image(request.rescan_image)
        height, width = image.shape[:2]
        
        # Convert data
        expected_point = _dict_to_point(request.expected_point)
        bbox = _dict_to_bbox(request.bounding_box)
        
        # Detect mark
        physical_point = point_validation.detect_physical_mark(
            image, expected_point, bbox, width, height
        )
        
        # Validate
        result = point_validation.validate_point(expected_point, physical_point, width, height)
        
        # Generate message
        message = f"Move marker {result.adjustment.magnitude:.3f} units {result.adjustment.direction.replace('adjust_', '')}" if result.adjustment else "Point is correct"
        
        return FeedbackResponse(
            success=True,
            status=result.feedback,
            message=message,
            offset={
                "x": result.offset.x,
                "y": result.offset.y,
                "distance": result.offset.distance
            },
            adjustment={
                "direction": result.adjustment.direction,
                "magnitude": result.adjustment.magnitude,
                "units": result.adjustment.units
            } if result.adjustment else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback generation failed: {str(e)}")


@router.post("/symmetry-map", response_model=SymmetryMapResponse)
async def symmetry_map(request: SymmetryMapRequest):
    """Map piercing point from first ear to second ear."""
    try:
        # Process images
        ear1_image = image_processor.process_image(request.ear1_image)
        ear2_image = image_processor.process_image(request.ear2_image)
        
        height1, width1 = ear1_image.shape[:2]
        height2, width2 = ear2_image.shape[:2]
        
        # Detect and measure both ears
        ear1_result = ear_detection.detect_ear(ear1_image)
        ear2_result = ear_detection.detect_ear(ear2_image)
        
        if not ear1_result.ear_detected or not ear2_result.ear_detected:
            raise HTTPException(status_code=400, detail="Both ears must be detected")
        
        ear1_dimensions = ear_measurement.measure_dimensions(ear1_result, width1, height1)
        ear2_dimensions = ear_measurement.measure_dimensions(ear2_result, width2, height2)
        
        # Convert point
        ear1_point = _dict_to_point(request.ear1_point)
        
        # Map point
        mapped_point = symmetry_mapping.map_point_to_second_ear(
            ear1_point, ear1_dimensions, ear2_dimensions
        )
        
        # Calculate scale factors
        scale_factors = symmetry_mapping.calculate_scale_factor(
            ear1_dimensions, ear2_dimensions
        )
        
        return SymmetryMapResponse(
            success=True,
            mapped_point={"x": mapped_point.x, "y": mapped_point.y},
            scale_factor=scale_factors["scale_factor"],
            scale_x=scale_factors["scale_x"],
            scale_y=scale_factors["scale_y"],
            ear2_dimensions={
                "length": ear2_dimensions.length,
                "width": ear2_dimensions.width,
                "length_pixels": ear2_dimensions.length_pixels,
                "width_pixels": ear2_dimensions.width_pixels
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symmetry mapping failed: {str(e)}")


@router.post("/rescan-validate", response_model=RescanValidateResponse)
async def rescan_validate(request: RescanValidateRequest):
    """Validate with re-scan and track history."""
    try:
        # Process image
        image = image_processor.process_image(request.rescan_image)
        height, width = image.shape[:2]
        
        # Convert data
        original_point = _dict_to_point(request.original_point)
        bbox = _dict_to_bbox(request.bounding_box)
        
        # Convert validation history
        history = []
        for h in request.validation_history:
            history.append(ValidationResult(
                valid=h.get("valid", False),
                offset=Offset(
                    x=h.get("offset", {}).get("x", 0),
                    y=h.get("offset", {}).get("y", 0),
                    distance=h.get("offset", {}).get("distance", 0)
                ),
                feedback=h.get("feedback", ""),
                confidence=h.get("confidence", 0)
            ))
        
        # Validate
        result = point_validation.rescan_validate(
            image, original_point, history, bbox, width, height
        )
        
        return RescanValidateResponse(
            success=True,
            validated=result.best_result.valid if result.best_result else False,
            iterations=result.total_iterations,
            final_offset={
                "x": result.best_result.offset.x if result.best_result else 0,
                "y": result.best_result.offset.y if result.best_result else 0,
                "distance": result.best_result.offset.distance if result.best_result else 0
            },
            best_result={
                "valid": result.best_result.valid,
                "offset": {
                    "x": result.best_result.offset.x,
                    "y": result.best_result.offset.y,
                    "distance": result.best_result.offset.distance
                },
                "feedback": result.best_result.feedback,
                "confidence": result.best_result.confidence
            } if result.best_result else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-scan validation failed: {str(e)}")

