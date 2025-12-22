"""
Tests for ear measurement service.
"""
import pytest
from src.services.ear_measurement import EarMeasurementService
from src.models.ear import EarDetectionResult, Landmark, BoundingBox


def test_ear_measurement_service_init():
    """Test ear measurement service initialization."""
    service = EarMeasurementService()
    assert service is not None


def test_calculate_scale_factor():
    """Test scale factor calculation."""
    service = EarMeasurementService()
    
    from src.models.ear import EarDimensions
    
    ear1 = EarDimensions(length=0.5, width=0.3)
    ear2 = EarDimensions(length=0.6, width=0.36)
    
    scale_x, scale_y = service.calculate_scale_factor(ear1, ear2)
    assert scale_x == pytest.approx(1.2, rel=0.01)
    assert scale_y == pytest.approx(1.2, rel=0.01)

