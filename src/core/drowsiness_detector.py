"""
Main Drowsiness Detector Module

Orchestrates all detection components into a unified drowsiness detection system.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import yaml

from src.detection import FaceDetector, EyeTracker, YawnDetector, HeadPoseEstimator
from src.core.alert_system import AlertSystem
from src.utils.visualization import (
    draw_drowsiness_overlay,
    draw_eye_contours,
    draw_mouth_contour,
    draw_landmarks,
    draw_fps
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for all detection results from a single frame."""
    
    # Detection status
    face_detected: bool = False
    
    # Eye metrics
    ear: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    eyes_closed: bool = False
    blink_count: int = 0
    
    # Yawn metrics
    mar: float = 0.0
    is_yawning: bool = False
    yawn_count: int = 0
    
    # Head pose
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    head_down: bool = False
    head_tilted: bool = False
    
    # Overall drowsiness
    drowsiness_score: float = 0.0
    is_drowsy: bool = False
    is_fatigued: bool = False
    
    # Landmarks (for visualization)
    landmarks: Optional[np.ndarray] = None
    
    # Timing
    processing_time_ms: float = 0.0


@dataclass
class Config:
    """Configuration container for drowsiness detector."""
    
    # Detection thresholds
    ear_threshold: float = 0.25
    mar_threshold: float = 0.75
    drowsy_time_seconds: float = 2.0
    blink_time_ms: float = 100
    yawn_consecutive_threshold: int = 3
    head_pose_threshold: float = 30
    
    # Camera settings
    camera_device_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    face_downsample_ratio: float = 1.5
    
    # Alert settings
    sound_enabled: bool = True
    sound_file: str = "alarm.wav"
    visual_enabled: bool = True
    alert_cooldown_seconds: float = 3.0
    
    # Model settings
    face_landmarks_path: str = "models/shape_predictor_68_face_landmarks.dat"
    model_download_url: str = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            # Detection
            ear_threshold=data.get('detection', {}).get('ear_threshold', 0.25),
            mar_threshold=data.get('detection', {}).get('mar_threshold', 0.75),
            drowsy_time_seconds=data.get('detection', {}).get('drowsy_time_seconds', 2.0),
            blink_time_ms=data.get('detection', {}).get('blink_time_ms', 100),
            yawn_consecutive_threshold=data.get('detection', {}).get('yawn_consecutive_threshold', 3),
            head_pose_threshold=data.get('detection', {}).get('head_pose_threshold', 30),
            
            # Camera
            camera_device_id=data.get('camera', {}).get('device_id', 0),
            camera_width=data.get('camera', {}).get('width', 640),
            camera_height=data.get('camera', {}).get('height', 480),
            camera_fps=data.get('camera', {}).get('fps', 30),
            face_downsample_ratio=data.get('camera', {}).get('face_downsample_ratio', 1.5),
            
            # Alerts
            sound_enabled=data.get('alerts', {}).get('sound_enabled', True),
            sound_file=data.get('alerts', {}).get('sound_file', 'assets/alarm.wav'),
            visual_enabled=data.get('alerts', {}).get('visual_enabled', True),
            alert_cooldown_seconds=data.get('alerts', {}).get('cooldown_seconds', 3.0),
            
            # Model
            face_landmarks_path=data.get('model', {}).get('face_landmarks', 'models/shape_predictor_68_face_landmarks.dat'),
            model_download_url=data.get('model', {}).get('download_url', 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'),
            
            # Logging
            log_level=data.get('logging', {}).get('level', 'INFO'),
        )


class DrowsinessDetector:
    """
    Main drowsiness detection orchestrator.
    
    Combines face detection, eye tracking, yawn detection, and head pose
    estimation to provide comprehensive drowsiness monitoring.
    
    Attributes:
        config: Configuration object
        face_detector: Face detection module
        eye_tracker: Eye tracking module
        yawn_detector: Yawn detection module
        head_pose: Head pose estimation module
        alert_system: Alert system for notifications
    """
    
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        """
        Initialize drowsiness detector.
        
        Args:
            config: Configuration object (takes precedence)
            config_path: Path to YAML config file
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None and Path(config_path).exists():
            self.config = Config.from_yaml(config_path)
        else:
            self.config = Config()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Initializing Drowsiness Detector...")
        
        # Calculate frame-based thresholds
        fps = self.config.camera_fps
        self.drowsy_frames_threshold = int(self.config.drowsy_time_seconds * fps)
        self.blink_frames_threshold = int((self.config.blink_time_ms / 1000) * fps)
        
        # Initialize components
        self.face_detector = FaceDetector(
            model_path=self.config.face_landmarks_path,
            downsample_ratio=self.config.face_downsample_ratio,
            download_url=self.config.model_download_url
        )
        
        self.eye_tracker = EyeTracker(
            ear_threshold=self.config.ear_threshold,
            drowsy_frames_threshold=self.drowsy_frames_threshold
        )
        
        self.yawn_detector = YawnDetector(
            mar_threshold=self.config.mar_threshold,
            consecutive_yawns_alert=self.config.yawn_consecutive_threshold
        )
        
        self.head_pose = HeadPoseEstimator(
            pitch_threshold=self.config.head_pose_threshold,
            frame_width=self.config.camera_width,
            frame_height=self.config.camera_height
        )
        
        self.alert_system = AlertSystem(
            sound_enabled=self.config.sound_enabled,
            sound_file=self.config.sound_file,
            visual_enabled=self.config.visual_enabled,
            cooldown_seconds=self.config.alert_cooldown_seconds
        )
        
        # State
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Drowsiness Detector initialized successfully")
    
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single frame and detect drowsiness indicators.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            DetectionResult with all detection metrics
        """
        start_time = time.time()
        self.frame_count += 1
        
        result = DetectionResult()
        
        # Get facial landmarks
        landmarks = self.face_detector.get_landmarks(frame)
        
        if landmarks is None:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        result.face_detected = True
        result.landmarks = landmarks
        
        # Eye tracking
        left_eye, right_eye = self.face_detector.get_eye_landmarks(landmarks)
        eye_metrics = self.eye_tracker.process_eyes(left_eye, right_eye)
        
        result.ear = eye_metrics.avg_ear
        result.left_ear = eye_metrics.left_ear
        result.right_ear = eye_metrics.right_ear
        result.eyes_closed = eye_metrics.is_closed
        result.blink_count = self.eye_tracker.blink_count
        
        # Yawn detection
        mouth_landmarks = self.face_detector.get_mouth_landmarks(landmarks)
        yawn_metrics = self.yawn_detector.process_mouth(mouth_landmarks)
        
        result.mar = yawn_metrics.mar
        result.is_yawning = yawn_metrics.is_yawning
        result.yawn_count = yawn_metrics.total_yawns
        
        # Head pose estimation
        head_metrics = self.head_pose.estimate_pose(landmarks, frame.shape[:2])
        
        result.pitch = head_metrics.pitch
        result.yaw = head_metrics.yaw
        result.roll = head_metrics.roll
        result.head_down = head_metrics.is_head_down
        result.head_tilted = head_metrics.is_head_tilted
        
        # Calculate overall drowsiness
        eye_drowsiness = self.eye_tracker.get_drowsiness_score()
        yawn_fatigue = 30.0 if self.yawn_detector.is_fatigue_indicated() else 0.0
        head_drowsiness = 20.0 if head_metrics.is_head_down else 0.0
        
        result.drowsiness_score = min(eye_drowsiness + yawn_fatigue + head_drowsiness, 100.0)
        result.is_drowsy = self.eye_tracker.is_drowsy() or head_metrics.is_head_down
        result.is_fatigued = self.yawn_detector.is_fatigue_indicated()
        
        # Trigger alerts if needed
        if result.is_drowsy:
            self.alert_system.trigger_alert("drowsiness")
        elif result.is_fatigued:
            self.alert_system.trigger_alert("fatigue")
        elif not result.eyes_closed and not result.is_yawning:
            self.alert_system.stop_alert()
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def draw_visualization(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        show_landmarks: bool = True,
        show_contours: bool = True
    ) -> np.ndarray:
        """
        Draw visualization overlays on frame.
        
        Args:
            frame: BGR image
            result: Detection result from process_frame
            show_landmarks: Whether to draw facial landmarks
            show_contours: Whether to draw eye/mouth contours
            
        Returns:
            Frame with visualizations
        """
        frame = frame.copy()
        
        if result.landmarks is not None:
            # Draw landmarks
            if show_landmarks:
                frame = draw_landmarks(frame, result.landmarks, radius=1)
            
            # Draw contours
            if show_contours:
                left_eye, right_eye = self.face_detector.get_eye_landmarks(result.landmarks)
                frame = draw_eye_contours(frame, left_eye, right_eye)
                
                mouth = self.face_detector.get_mouth_landmarks(result.landmarks)
                frame = draw_mouth_contour(frame, mouth)
        
        # Draw status overlay
        frame = draw_drowsiness_overlay(
            frame,
            is_drowsy=result.is_drowsy,
            drowsiness_score=result.drowsiness_score,
            blink_count=result.blink_count,
            ear_value=result.ear,
            is_yawning=result.is_yawning,
            yawn_count=result.yawn_count,
            mar_value=result.mar,
            head_pose=(result.pitch, result.yaw, result.roll)
        )
        
        # Draw FPS
        fps = self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0
        frame = draw_fps(frame, fps)
        
        return frame
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.eye_tracker.reset()
        self.yawn_detector.reset()
        self.alert_system.reset()
        self.frame_count = 0
        self.start_time = time.time()
        logger.info("Detector reset")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.alert_system.cleanup()
        logger.info("Detector cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with stats
        """
        elapsed = time.time() - self.start_time
        return {
            'frames_processed': self.frame_count,
            'elapsed_seconds': elapsed,
            'avg_fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'blink_count': self.eye_tracker.blink_count,
            'yawn_count': self.yawn_detector.total_yawns,
            'alert_count': self.alert_system.alert_count,
        }
