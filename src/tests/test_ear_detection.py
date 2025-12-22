"""
Tests for ear detection service.
"""
import pytest
import numpy as np
from src.services.ear_detection import EarDetectionService


def test_ear_detection_service_init():
    """Test ear detection service initialization."""
    service = EarDetectionService()
    assert service is not None
    assert service.settings is not None


def test_ear_detection_no_ear():
    """Test ear detection with no ear present."""
    service = EarDetectionService()
    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = service.detect_ear(image)
    # Should handle gracefully even if no ear detected
    assert result is not None

