"""
Head Pose Estimation Module

Estimates 3D head orientation to detect nodding, tilting, and head drooping.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeadPoseMetrics:
    """Container for head pose metrics."""
    pitch: float  # Up/down rotation (nodding)
    yaw: float    # Left/right rotation
    roll: float   # Tilt (ear to shoulder)
    is_head_down: bool
    is_head_tilted: bool
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None


class HeadPoseEstimator:
    """
    Head pose estimator using solvePnP algorithm.
    
    Uses 6 facial landmarks to estimate 3D head orientation:
    - Nose tip
    - Chin
    - Left eye corner
    - Right eye corner
    - Left mouth corner
    - Right mouth corner
    
    Attributes:
        pitch_threshold: Degrees of pitch to consider head down
        roll_threshold: Degrees of roll to consider head tilted
    """
    
    # 3D model points of a generic face (from Anthropometric data)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    # Indices in 68-point landmark model
    LANDMARK_INDICES = {
        'nose_tip': 30,
        'chin': 8,
        'left_eye_corner': 36,
        'right_eye_corner': 45,
        'left_mouth_corner': 48,
        'right_mouth_corner': 54
    }
    
    def __init__(
        self,
        pitch_threshold: float = 15.0,
        roll_threshold: float = 20.0,
        frame_width: int = 640,
        frame_height: int = 480
    ):
        """
        Initialize head pose estimator.
        
        Args:
            pitch_threshold: Degrees below which head is considered down
            roll_threshold: Degrees beyond which head is considered tilted
            frame_width: Width of video frame for camera matrix
            frame_height: Height of video frame for camera matrix
        """
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        
        # Camera internals (approximate for webcam)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._update_camera_matrix()
        
        # Distortion coefficients (assuming no lens distortion)
        self.dist_coeffs = np.zeros((4, 1))
        
        logger.info(f"Head pose estimator initialized "
                   f"(pitch_thresh={pitch_threshold}°, roll_thresh={roll_threshold}°)")
    
    def _update_camera_matrix(self) -> None:
        """Update camera matrix based on frame dimensions."""
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def set_frame_size(self, width: int, height: int) -> None:
        """Update frame size and recalculate camera matrix."""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self._update_camera_matrix()
    
    def _extract_pose_points(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract the 6 points needed for pose estimation from 68 landmarks.
        
        Args:
            landmarks: (68, 2) array of facial landmarks
            
        Returns:
            (6, 2) array of pose estimation points
        """
        indices = [
            self.LANDMARK_INDICES['nose_tip'],
            self.LANDMARK_INDICES['chin'],
            self.LANDMARK_INDICES['left_eye_corner'],
            self.LANDMARK_INDICES['right_eye_corner'],
            self.LANDMARK_INDICES['left_mouth_corner'],
            self.LANDMARK_INDICES['right_mouth_corner']
        ]
        
        return landmarks[indices].astype(np.float64)
    
    def estimate_pose(
        self,
        landmarks: np.ndarray,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> HeadPoseMetrics:
        """
        Estimate head pose from facial landmarks.
        
        Args:
            landmarks: (68, 2) array of facial landmarks
            frame_shape: Optional (height, width) to update camera matrix
            
        Returns:
            HeadPoseMetrics with rotation angles and flags
        """
        # Update camera matrix if frame size changed
        if frame_shape is not None:
            self.set_frame_size(frame_shape[1], frame_shape[0])
        
        # Extract the 6 points for pose estimation
        image_points = self._extract_pose_points(landmarks)
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            logger.warning("Pose estimation failed")
            return HeadPoseMetrics(
                pitch=0.0, yaw=0.0, roll=0.0,
                is_head_down=False, is_head_tilted=False
            )
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles from rotation matrix
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
        
        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        
        # Check thresholds
        is_head_down = pitch_deg < -self.pitch_threshold
        is_head_tilted = abs(roll_deg) > self.roll_threshold
        
        return HeadPoseMetrics(
            pitch=pitch_deg,
            yaw=yaw_deg,
            roll=roll_deg,
            is_head_down=is_head_down,
            is_head_tilted=is_head_tilted,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector
        )
    
    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) in radians
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])   # Roll
            y = np.arctan2(-R[2, 0], sy)        # Pitch
            z = np.arctan2(R[1, 0], R[0, 0])   # Yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return y, z, x  # pitch, yaw, roll
    
    def draw_pose_axes(
        self,
        frame: np.ndarray,
        rotation_vector: np.ndarray,
        translation_vector: np.ndarray,
        landmarks: np.ndarray,
        axis_length: float = 100.0
    ) -> np.ndarray:
        """
        Draw 3D pose axes on the frame.
        
        Args:
            frame: BGR image to draw on
            rotation_vector: Rotation vector from solvePnP
            translation_vector: Translation vector from solvePnP
            landmarks: Facial landmarks for nose position
            axis_length: Length of axes to draw
            
        Returns:
            Frame with axes drawn
        """
        # Get nose tip as origin
        nose_tip = tuple(landmarks[self.LANDMARK_INDICES['nose_tip']].astype(int))
        
        # Define axis endpoints in 3D
        axes_3d = np.array([
            [axis_length, 0, 0],   # X axis (red)
            [0, axis_length, 0],   # Y axis (green)
            [0, 0, axis_length]    # Z axis (blue)
        ], dtype=np.float64)
        
        # Project to 2D
        axes_2d, _ = cv2.projectPoints(
            axes_3d,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # Draw axes
        frame = frame.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB -> BGR
        
        for axis_point, color in zip(axes_2d, colors):
            end_point = tuple(axis_point.ravel().astype(int))
            cv2.line(frame, nose_tip, end_point, color, 2)
        
        return frame
