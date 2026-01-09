"""
Face Detection Module

Provides face detection using dlib's HOG-based detector with facial landmark prediction.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import bz2
import urllib.request

import cv2
import dlib
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using dlib's frontal face detector and 68-point landmark predictor.
    
    Attributes:
        detector: dlib HOG-based face detector
        predictor: dlib facial landmark predictor
        downsample_ratio: Factor to downsample image for faster detection
    """
    
    # Landmark indices for different facial features
    LANDMARKS = {
        "jaw": list(range(0, 17)),
        "right_eyebrow": list(range(17, 22)),
        "left_eyebrow": list(range(22, 27)),
        "nose": list(range(27, 36)),
        "right_eye": list(range(36, 42)),
        "left_eye": list(range(42, 48)),
        "outer_mouth": list(range(48, 60)),
        "inner_mouth": list(range(60, 68)),
    }
    
    def __init__(
        self,
        model_path: str = "models/shape_predictor_68_face_landmarks.dat",
        downsample_ratio: float = 1.5,
        download_url: str = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    ):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to the dlib shape predictor model
            downsample_ratio: Factor to downsample image for faster detection
            download_url: URL to download model if not found
        """
        self.downsample_ratio = downsample_ratio
        self.model_path = Path(model_path)
        self.download_url = download_url
        
        # Ensure model exists
        self._ensure_model_exists()
        
        # Initialize dlib detector and predictor
        logger.info("Initializing dlib face detector...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(self.model_path))
        logger.info("Face detector initialized successfully")
    
    def _ensure_model_exists(self) -> None:
        """Download the model file if it doesn't exist."""
        if self.model_path.exists():
            return
            
        logger.info(f"Model not found at {self.model_path}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        bz2_path = self.model_path.with_suffix(".dat.bz2")
        
        logger.info(f"Downloading model from {self.download_url}...")
        try:
            urllib.request.urlretrieve(self.download_url, bz2_path)
            
            logger.info("Extracting model...")
            with bz2.open(bz2_path, 'rb') as src:
                with open(self.model_path, 'wb') as dst:
                    dst.write(src.read())
            
            bz2_path.unlink()  # Remove compressed file
            logger.info("Model downloaded and extracted successfully")
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise RuntimeError(
                f"Could not download facial landmark model. "
                f"Please manually download from {self.download_url} "
                f"and extract to {self.model_path}"
            )
    
    def detect_faces(self, frame: np.ndarray) -> List[dlib.rectangle]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of dlib rectangles representing detected faces
        """
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Downsample for faster detection
        small = cv2.resize(
            gray,
            None,
            fx=1.0 / self.downsample_ratio,
            fy=1.0 / self.downsample_ratio,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Detect faces
        rects = self.detector(small, 0)
        
        # Scale rectangles back to original size
        scaled_rects = []
        for rect in rects:
            scaled_rect = dlib.rectangle(
                int(rect.left() * self.downsample_ratio),
                int(rect.top() * self.downsample_ratio),
                int(rect.right() * self.downsample_ratio),
                int(rect.bottom() * self.downsample_ratio)
            )
            scaled_rects.append(scaled_rect)
        
        return scaled_rects
    
    def get_landmarks(
        self,
        frame: np.ndarray,
        face_rect: Optional[dlib.rectangle] = None
    ) -> Optional[np.ndarray]:
        """
        Get 68 facial landmarks for the given face.
        
        Args:
            frame: BGR image from OpenCV
            face_rect: dlib rectangle of face. If None, detect face first.
            
        Returns:
            (68, 2) numpy array of landmark coordinates, or None if no face detected
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect face if not provided
        if face_rect is None:
            faces = self.detect_faces(frame)
            if not faces:
                return None
            face_rect = faces[0]
        
        # Get landmarks
        shape = self.predictor(gray, face_rect)
        
        # Convert to numpy array
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
        return landmarks
    
    def get_eye_landmarks(
        self,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from full facial landmarks.
        
        Args:
            landmarks: (68, 2) array of facial landmarks
            
        Returns:
            Tuple of (left_eye, right_eye) landmark arrays
        """
        left_eye = landmarks[self.LANDMARKS["left_eye"]]
        right_eye = landmarks[self.LANDMARKS["right_eye"]]
        return left_eye, right_eye
    
    def get_mouth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract mouth landmarks from full facial landmarks.
        
        Args:
            landmarks: (68, 2) array of facial landmarks
            
        Returns:
            Combined inner and outer mouth landmarks
        """
        outer = landmarks[self.LANDMARKS["outer_mouth"]]
        inner = landmarks[self.LANDMARKS["inner_mouth"]]
        return np.vstack([outer, inner])
    
    def get_face_rect_center(self, face_rect: dlib.rectangle) -> Tuple[int, int]:
        """Get the center point of a face rectangle."""
        x = (face_rect.left() + face_rect.right()) // 2
        y = (face_rect.top() + face_rect.bottom()) // 2
        return x, y
