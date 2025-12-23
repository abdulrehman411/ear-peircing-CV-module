"""
MediaPipe integration for ear landmark detection.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple
from threading import Lock
from src.config import get_settings
from src.models.ear import Landmark


class MediaPipeWrapper:
    """Wrapper for MediaPipe Face Mesh for ear detection with singleton pattern."""
    
    # MediaPipe face mesh landmark indices for ears
    LEFT_EAR_INDICES = list(range(234, 455))  # Left ear landmarks
    RIGHT_EAR_INDICES = []  # Will be calculated based on face orientation
    
    _instance: Optional['MediaPipeWrapper'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh model (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return
        
        self.settings = get_settings()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        self._initialized = False
        self._initialize()
    
    def _initialize(self):
        """Initialize MediaPipe Face Mesh model."""
        if self._initialized:
            return
            
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.settings.mediapipe_confidence_threshold,
                min_tracking_confidence=0.5
            )
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe: {str(e)}")
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[List[Landmark]]:
        """
        Detect face landmarks from image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of landmarks or None if no face detected
        """
        if self.face_mesh is None:
            self._initialize()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract all landmarks
        landmarks = []
        height, width = image.shape[:2]
        
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks.append(Landmark(
                x=landmark.x * width,
                y=landmark.y * height,
                z=landmark.z * width,  # Z is relative to width
                visibility=getattr(landmark, 'visibility', 1.0),
                index=idx
            ))
        
        return landmarks
    
    def extract_ear_landmarks(self, image: np.ndarray, ear_side: str = "left") -> Optional[List[Landmark]]:
        """
        Extract ear landmarks from image.
        
        Args:
            image: Input image
            ear_side: "left" or "right"
            
        Returns:
            List of ear landmarks or None
        """
        all_landmarks = self.detect_landmarks(image)
        
        if all_landmarks is None:
            return None
        
        # Determine which ear to extract based on face orientation
        # For simplicity, we'll use left ear indices and detect orientation
        if ear_side == "left":
            indices = self.LEFT_EAR_INDICES
        else:
            # For right ear, we need to mirror or use different approach
            # MediaPipe landmarks are symmetric, so we can use mirrored indices
            # Right ear is typically on the opposite side
            indices = self.LEFT_EAR_INDICES
        
        ear_landmarks = [all_landmarks[i] for i in indices if i < len(all_landmarks)]
        
        return ear_landmarks if ear_landmarks else None
    
    def detect_ear_side(self, image: np.ndarray) -> Optional[str]:
        """
        Detect which ear is visible (left or right).
        
        Args:
            image: Input image
            
        Returns:
            "left" or "right" or None
        """
        landmarks = self.detect_landmarks(image)
        
        if landmarks is None or len(landmarks) < 10:
            return None
        
        # Use nose tip and face center to determine orientation
        # Landmark 1 is nose tip, landmark 0 is face center
        if len(landmarks) > 1:
            # Simple heuristic: check if left ear landmarks are more visible
            left_ear_landmarks = [landmarks[i] for i in self.LEFT_EAR_INDICES if i < len(landmarks)]
            
            if left_ear_landmarks:
                # Check average visibility
                avg_visibility = sum(l.visibility or 0 for l in left_ear_landmarks) / len(left_ear_landmarks)
                
                if avg_visibility > 0.5:
                    return "left"
                else:
                    return "right"
        
        return "left"  # Default to left
    
    def get_ear_bounding_box(self, ear_landmarks: List[Landmark]) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculate bounding box for ear landmarks.
        
        Args:
            ear_landmarks: List of ear landmarks
            
        Returns:
            Bounding box (x_min, y_min, x_max, y_max) or None
        """
        if not ear_landmarks:
            return None
        
        x_coords = [l.x for l in ear_landmarks]
        y_coords = [l.y for l in ear_landmarks]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        return (x_min, y_min, x_max, y_max)

