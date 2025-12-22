"""
Tests for symmetry mapping service.
"""
import pytest
from src.services.symmetry_mapping import SymmetryMappingService
from src.models.ear import Point, EarDimensions


def test_symmetry_mapping_service_init():
    """Test symmetry mapping service initialization."""
    service = SymmetryMappingService()
    assert service is not None


def test_map_point_to_second_ear():
    """Test point mapping to second ear."""
    service = SymmetryMappingService()
    
    ear1_point = Point(x=0.5, y=0.6)
    ear1_dimensions = EarDimensions(length=0.5, width=0.3)
    ear2_dimensions = EarDimensions(length=0.6, width=0.36)
    
    mapped = service.map_point_to_second_ear(
        ear1_point, ear1_dimensions, ear2_dimensions
    )
    
    assert mapped is not None
    assert 0.0 <= mapped.x <= 1.0
    assert 0.0 <= mapped.y <= 1.0


def test_calculate_scale_factor():
    """Test scale factor calculation."""
    service = SymmetryMappingService()
    
    ear1 = EarDimensions(length=0.5, width=0.3)
    ear2 = EarDimensions(length=0.6, width=0.36)
    
    factors = service.calculate_scale_factor(ear1, ear2)
    assert "scale_x" in factors
    assert "scale_y" in factors
    assert "scale_factor" in factors

