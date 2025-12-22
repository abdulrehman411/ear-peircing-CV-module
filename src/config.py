"""
Configuration management for the CV module.
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    env: str = "development"
    log_level: str = "INFO"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "v1"
    
    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:19006"
    
    # MediaPipe Configuration
    mediapipe_model_path: str = ""
    mediapipe_confidence_threshold: float = 0.5
    
    # Validation Configuration
    validation_threshold: float = 0.01  # Normalized units
    validation_max_iterations: int = 5
    
    # Image Processing Configuration
    max_image_size: int = 5000  # pixels
    image_preprocessing_enabled: bool = True
    lighting_correction_enabled: bool = True
    
    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    
    # Ear Detection Configuration
    ear_landmark_indices_left: List[int] = list(range(234, 455))  # MediaPipe left ear landmarks
    ear_landmark_indices_right: List[int] = []  # Will be calculated as mirrored
    
    # Piercing Detection Configuration
    piercing_detection_confidence: float = 0.7
    piercing_color_ranges: dict = {
        "gold": {"lower": [20, 100, 100], "upper": [30, 255, 255]},
        "silver": {"lower": [0, 0, 100], "upper": [180, 30, 255]}
    }
    
    # Mark Detection Configuration
    mark_color_ranges: dict = {
        "blue": {"lower": [100, 50, 50], "upper": [130, 255, 255]},
        "black": {"lower": [0, 0, 0], "upper": [180, 255, 50]}
    }
    mark_min_area: int = 10  # pixels
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    # Calculate right ear landmarks as mirrored indices
    if not settings.ear_landmark_indices_right:
        # MediaPipe has 468 landmarks, right ear is mirrored
        # For simplicity, we'll use the same indices but detect orientation
        settings.ear_landmark_indices_right = settings.ear_landmark_indices_left
    return settings

