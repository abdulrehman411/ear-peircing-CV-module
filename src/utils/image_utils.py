"""
Image manipulation utilities.
"""
import base64
import io
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 encoded image to numpy array.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        numpy array of image (BGR format for OpenCV)
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Convert numpy array image to base64 string.
    
    Args:
        image: numpy array image (BGR format)
        format: Image format (JPEG, PNG)
        
    Returns:
        Base64 encoded image string
    """
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string
    except Exception as e:
        raise ValueError(f"Failed to encode image to base64: {str(e)}")


def resize_image(image: np.ndarray, max_size: int, maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if maintain_aspect:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
    else:
        new_width = max_size
        new_height = max_size
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Normalize lighting using histogram equalization.
    
    Args:
        image: Input image
        
    Returns:
        Image with normalized lighting
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized


def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Reduce noise using Gaussian blur.
    
    Args:
        image: Input image
        kernel_size: Gaussian kernel size (must be odd)
        
    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return denoised


def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Enhanced image
    """
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced


def crop_region(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10) -> np.ndarray:
    """
    Crop image region with padding.
    
    Args:
        image: Input image
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        padding: Padding in pixels
        
    Returns:
        Cropped image
    """
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    # Add padding
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(width, int(x_max) + padding)
    y_max = min(height, int(y_max) + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

