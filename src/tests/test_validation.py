"""
Tests for point validation service.
"""
import pytest
from src.services.point_validation import PointValidationService
from src.models.ear import Point, BoundingBox, Landmark


def test_point_validation_service_init():
    """Test point validation service initialization."""
    service = PointValidationService()
    assert service is not None


def test_generate_feedback():
    """Test feedback generation."""
    service = PointValidationService()
    
    # Test correct point
    feedback, adjustment = service._generate_feedback(0.0, 0.0, 0.005)
    assert feedback == "correct"
    
    # Test horizontal offset
    feedback, adjustment = service._generate_feedback(0.02, 0.0, 0.02)
    assert "adjust" in feedback
    assert adjustment.magnitude > 0

