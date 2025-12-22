"""
Tests for coordinate transformation utilities.
"""
import pytest
from src.utils.coordinate_transform import (
    pixel_to_normalized,
    normalized_to_pixel,
    calculate_distance,
    mirror_point,
    apply_scaling
)
from src.models.ear import Point


def test_pixel_to_normalized():
    """Test pixel to normalized conversion."""
    x, y = pixel_to_normalized(100, 200, 1000, 2000)
    assert x == 0.1
    assert y == 0.1


def test_normalized_to_pixel():
    """Test normalized to pixel conversion."""
    x, y = normalized_to_pixel(0.1, 0.1, 1000, 2000)
    assert x == 100
    assert y == 200


def test_calculate_distance():
    """Test distance calculation."""
    p1 = Point(x=0, y=0)
    p2 = Point(x=3, y=4)
    distance = calculate_distance(p1, p2)
    assert distance == 5.0


def test_mirror_point():
    """Test point mirroring."""
    point = Point(x=0.3, y=0.5)
    mirrored = mirror_point(point, width=1.0)
    assert mirrored.x == 0.7
    assert mirrored.y == 0.5


def test_apply_scaling():
    """Test point scaling."""
    point = Point(x=0.5, y=0.5)
    scaled = apply_scaling(point, 2.0, 0.5)
    assert scaled.x == 1.0
    assert scaled.y == 0.25

