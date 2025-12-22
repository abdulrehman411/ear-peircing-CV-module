"""
Existing piercing detection service.
"""
import cv2
import numpy as np
from typing import List
from src.config import get_settings
from src.models.piercing import Piercing, PiercingDetectionResult
from src.models.ear import Point, Landmark, BoundingBox
from src.utils.coordinate_transform import pixel_to_normalized, to_ear_relative


class PiercingDetectionService:
    """Service for detecting existing piercings in ear images."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def detect_piercings(
        self,
        image: np.ndarray,
        ear_landmarks: List[Landmark],
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> PiercingDetectionResult:
        """
        Detect existing piercings in ear image.
        
        Args:
            image: Input image (BGR format)
            ear_landmarks: Ear landmarks
            bbox: Ear bounding box
            image_width: Image width
            image_height: Image height
            
        Returns:
            PiercingDetectionResult with detected piercings
        """
        piercings = []
        
        # Extract ear region
        ear_region = self._extract_ear_region(image, bbox)
        
        # Method 1: Color-based detection (gold, silver)
        color_piercings = self._detect_by_color(ear_region, bbox, image_width, image_height)
        piercings.extend(color_piercings)
        
        # Method 2: Blob detection for metallic objects
        blob_piercings = self._detect_by_blob(ear_region, bbox, image_width, image_height)
        piercings.extend(blob_piercings)
        
        # Method 3: Edge detection around potential locations
        edge_piercings = self._detect_by_edges(ear_region, bbox, image_width, image_height)
        piercings.extend(edge_piercings)
        
        # Remove duplicates and filter by confidence
        unique_piercings = self._filter_duplicates(piercings)
        
        return PiercingDetectionResult(
            piercings=unique_piercings,
            total_count=len(unique_piercings)
        )
    
    def _extract_ear_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extract ear region from image."""
        x_min = max(0, int(bbox.x_min))
        y_min = max(0, int(bbox.y_min))
        x_max = min(image.shape[1], int(bbox.x_max))
        y_max = min(image.shape[0], int(bbox.y_max))
        
        return image[y_min:y_max, x_min:x_max]
    
    def _detect_by_color(
        self,
        ear_region: np.ndarray,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> List[Piercing]:
        """Detect piercings by color (gold, silver)."""
        piercings = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(ear_region, cv2.COLOR_BGR2HSV)
        
        for color_name, color_range in self.settings.piercing_color_ranges.items():
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])
            
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Minimum area threshold
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Convert to image coordinates
                        img_x = bbox.x_min + cx
                        img_y = bbox.y_min + cy
                        
                        # Normalize
                        norm_x, norm_y = pixel_to_normalized(img_x, img_y, image_width, image_height)
                        
                        # Determine piercing type
                        piercing_type = "stud" if area < 100 else "hoop"
                        
                        piercings.append(Piercing(
                            point=Point(x=norm_x, y=norm_y),
                            type=piercing_type,
                            confidence=min(area / 200.0, 1.0),
                            size=area / (image_width * image_height)
                        ))
        
        return piercings
    
    def _detect_by_blob(
        self,
        ear_region: np.ndarray,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> List[Piercing]:
        """Detect piercings using blob detection."""
        piercings = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(ear_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Reasonable size for piercing
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.5:  # Relatively circular
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            img_x = bbox.x_min + cx
                            img_y = bbox.y_min + cy
                            
                            norm_x, norm_y = pixel_to_normalized(img_x, img_y, image_width, image_height)
                            
                            piercings.append(Piercing(
                                point=Point(x=norm_x, y=norm_y),
                                type="stud",
                                confidence=circularity,
                                size=area / (image_width * image_height)
                            ))
        
        return piercings
    
    def _detect_by_edges(
        self,
        ear_region: np.ndarray,
        bbox: BoundingBox,
        image_width: int,
        image_height: int
    ) -> List[Piercing]:
        """Detect piercings using edge detection."""
        piercings = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(ear_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find circles using HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, radius = circle
                
                img_x = bbox.x_min + cx
                img_y = bbox.y_min + cy
                
                norm_x, norm_y = pixel_to_normalized(img_x, img_y, image_width, image_height)
                
                piercings.append(Piercing(
                    point=Point(x=norm_x, y=norm_y),
                    type="hoop" if radius > 15 else "stud",
                    confidence=0.7,
                    size=radius / min(image_width, image_height)
                ))
        
        return piercings
    
    def _filter_duplicates(self, piercings: List[Piercing], threshold: float = 0.05) -> List[Piercing]:
        """Filter duplicate piercings based on proximity."""
        if not piercings:
            return []
        
        unique = []
        for piercing in piercings:
            is_duplicate = False
            for existing in unique:
                distance = np.sqrt(
                    (piercing.point.x - existing.point.x) ** 2 +
                    (piercing.point.y - existing.point.y) ** 2
                )
                if distance < threshold:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if piercing.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(piercing)
                    break
            
            if not is_duplicate:
                unique.append(piercing)
        
        return unique

