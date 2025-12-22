"""
Point validation and feedback service.
"""
import cv2
import numpy as np
import time
from typing import Optional, List
from src.config import get_settings
from src.models.validation import (
    ValidationResult,
    Offset,
    Adjustment,
    PointMark,
    ValidationHistory
)
from src.models.ear import Point, Landmark, BoundingBox
from src.utils.coordinate_transform import (
    calculate_distance,
    pixel_to_normalized,
    normalized_to_pixel
)


class PointValidationService:
    """Service for validating physical marks against digital points."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def mark_point(
        self,
        point: Point,
        ear_landmarks: List[Landmark],
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> PointMark:
        """
        Mark a digital point and store landmark references.
        
        Args:
            point: User-selected point (normalized)
            ear_landmarks: Ear landmarks
            bbox: Ear bounding box
            image_width: Image width
            image_height: Image height
            
        Returns:
            PointMark with landmark references
        """
        # Calculate distances to key landmarks
        landmark_references = {}
        
        # Convert point to pixel coordinates for calculations
        pixel_point = normalized_to_pixel(point.x, point.y, image_width, image_height)
        pixel_point_obj = Point(x=pixel_point[0], y=pixel_point[1])
        
        # Calculate distances to some key landmarks (helix, lobe, etc.)
        if ear_landmarks:
            # Use first, middle, and last landmarks as references
            key_indices = [0, len(ear_landmarks) // 2, len(ear_landmarks) - 1]
            
            for idx in key_indices:
                if idx < len(ear_landmarks):
                    landmark = ear_landmarks[idx]
                    landmark_pixel = normalized_to_pixel(
                        landmark.x, landmark.y, image_width, image_height
                    )
                    landmark_point = Point(x=landmark_pixel[0], y=landmark_pixel[1])
                    
                    distance = calculate_distance(pixel_point_obj, landmark_point)
                    landmark_references[f"landmark_{idx}"] = distance / min(image_width, image_height)
        
        return PointMark(
            point=point,
            landmark_references=landmark_references,
            normalized=True,
            timestamp=time.time()
        )
    
    def detect_physical_mark(
        self,
        image: np.ndarray,
        expected_point: Point,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> Optional[Point]:
        """
        Detect physical mark on ear image.
        
        Args:
            image: Re-scanned image with physical mark
            expected_point: Expected point location (normalized)
            bbox: Ear bounding box
            image_width: Image width
            image_height: Image height
            
        Returns:
            Detected mark point or None
        """
        # Extract ear region
        ear_region = self._extract_ear_region(image, bbox)
        
        # Method 1: Color detection (blue/black marker)
        mark_point = self._detect_mark_by_color(
            ear_region, expected_point, bbox, image_width, image_height
        )
        
        if mark_point:
            return mark_point
        
        # Method 2: Blob detection
        mark_point = self._detect_mark_by_blob(
            ear_region, expected_point, bbox, image_width, image_height
        )
        
        return mark_point
    
    def validate_point(
        self,
        digital_point: Point,
        physical_point: Optional[Point],
        image_width: int,
        image_height: int
    ) -> ValidationResult:
        """
        Validate physical point against digital point.
        
        Args:
            digital_point: Digital point (normalized)
            physical_point: Detected physical point (normalized) or None
            image_width: Image width
            image_height: Image height
            
        Returns:
            ValidationResult with feedback
        """
        if physical_point is None:
            return ValidationResult(
                valid=False,
                offset=Offset(x=0.0, y=0.0, distance=float('inf')),
                feedback="mark_not_detected",
                confidence=0.0
            )
        
        # Calculate offset
        offset_x = physical_point.x - digital_point.x
        offset_y = physical_point.y - digital_point.y
        distance = np.sqrt(offset_x ** 2 + offset_y ** 2)
        
        offset = Offset(x=offset_x, y=offset_y, distance=distance)
        
        # Check if within threshold
        is_valid = distance <= self.settings.validation_threshold
        
        # Generate feedback
        feedback, adjustment = self._generate_feedback(offset_x, offset_y, distance)
        
        return ValidationResult(
            valid=is_valid,
            offset=offset,
            feedback=feedback,
            adjustment=adjustment,
            confidence=1.0 - min(distance / self.settings.validation_threshold, 1.0)
        )
    
    def _extract_ear_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extract ear region from image."""
        x_min = max(0, int(bbox.x_min))
        y_min = max(0, int(bbox.y_min))
        x_max = min(image.shape[1], int(bbox.x_max))
        y_max = min(image.shape[0], int(bbox.y_max))
        
        return image[y_min:y_max, x_min:x_max]
    
    def _detect_mark_by_color(
        self,
        ear_region: np.ndarray,
        expected_point: Point,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> Optional[Point]:
        """Detect mark by color (blue/black)."""
        # Convert to HSV
        hsv = cv2.cvtColor(ear_region, cv2.COLOR_BGR2HSV)
        
        # Search around expected point
        expected_pixel = normalized_to_pixel(expected_point.x, expected_point.y, image_width, image_height)
        search_radius = 50  # pixels
        
        for color_name, color_range in self.settings.mark_color_ranges.items():
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.settings.mark_min_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if within search radius
                        dist = np.sqrt(
                            (cx - (expected_pixel[0] - bbox.x_min)) ** 2 +
                            (cy - (expected_pixel[1] - bbox.y_min)) ** 2
                        )
                        
                        if dist < search_radius:
                            img_x = bbox.x_min + cx
                            img_y = bbox.y_min + cy
                            
                            norm_x, norm_y = pixel_to_normalized(img_x, img_y, image_width, image_height)
                            return Point(x=norm_x, y=norm_y)
        
        return None
    
    def _detect_mark_by_blob(
        self,
        ear_region: np.ndarray,
        expected_point: Point,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> Optional[Point]:
        """Detect mark using blob detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(ear_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        expected_pixel = normalized_to_pixel(expected_point.x, expected_point.y, image_width, image_height)
        search_radius = 50
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.settings.mark_min_area <= area <= 500:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    dist = np.sqrt(
                        (cx - (expected_pixel[0] - bbox.x_min)) ** 2 +
                        (cy - (expected_pixel[1] - bbox.y_min)) ** 2
                    )
                    
                    if dist < search_radius:
                        img_x = bbox.x_min + cx
                        img_y = bbox.y_min + cy
                        
                        norm_x, norm_y = pixel_to_normalized(img_x, img_y, image_width, image_height)
                        return Point(x=norm_x, y=norm_y)
        
        return None
    
    def _generate_feedback(self, offset_x: float, offset_y: float, distance: float) -> tuple:
        """Generate feedback message and adjustment."""
        threshold = self.settings.validation_threshold
        
        if distance <= threshold:
            return ("correct", Adjustment(direction="correct", magnitude=0.0, units="normalized"))
        
        # Determine primary direction
        abs_x = abs(offset_x)
        abs_y = abs(offset_y)
        
        if abs_x > abs_y:
            direction = "adjust_right" if offset_x > 0 else "adjust_left"
            magnitude = abs_x
        else:
            direction = "adjust_down" if offset_y > 0 else "adjust_up"
            magnitude = abs_y
        
        # Add secondary direction if significant
        if abs_x > threshold * 0.5 and abs_y > threshold * 0.5:
            if abs_x > abs_y:
                direction += "_and_" + ("down" if offset_y > 0 else "up")
            else:
                direction = ("right" if offset_x > 0 else "left") + "_and_" + direction.replace("adjust_", "")
        
        return (direction, Adjustment(direction=direction, magnitude=magnitude, units="normalized"))
    
    def rescan_validate(
        self,
        rescan_image: np.ndarray,
        original_point: Point,
        validation_history: List[ValidationResult],
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> ValidationHistory:
        """
        Validate with re-scan and track history.
        
        Args:
            rescan_image: Re-scanned image
            original_point: Original digital point
            validation_history: Previous validation results
            bbox: Ear bounding box
            image_width: Image width
            image_height: Image height
            
        Returns:
            ValidationHistory with updated results
        """
        # Detect physical mark
        physical_point = self.detect_physical_mark(
            rescan_image, original_point, bbox, image_width, image_height
        )
        
        # Validate
        result = self.validate_point(original_point, physical_point, image_width, image_height)
        
        # Update history
        history = ValidationHistory(iterations=validation_history + [result])
        history.total_iterations = len(history.iterations)
        
        # Find best result
        if history.iterations:
            history.best_result = min(
                history.iterations,
                key=lambda r: r.offset.distance if r.offset else float('inf')
            )
        
        return history

