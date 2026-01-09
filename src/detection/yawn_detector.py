"""
Yawn Detection Module

Implements Mouth Aspect Ratio (MAR) calculation for yawn detection.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)


@dataclass
class YawnMetrics:
    """Container for yawn-related metrics."""
    mar: float
    is_yawning: bool
    yawn_duration_frames: int
    total_yawns: int


class YawnDetector:
    """
    Yawn detector using Mouth Aspect Ratio (MAR) algorithm.
    
    Similar to EAR but for the mouth. High MAR indicates open mouth (potential yawn).
    
    The MAR formula uses vertical and horizontal mouth distances to create a ratio
    that's high when the mouth is open wide (yawning) and low when closed.
    
    Attributes:
        mar_threshold: MAR value above which mouth is considered in yawn position
        min_yawn_frames: Minimum consecutive frames for a valid yawn
    """
    
    # Mouth landmark indices (for 68-point model)
    # Outer mouth: 48-59
    # Inner mouth: 60-67
    OUTER_MOUTH_INDICES = list(range(48, 60))
    INNER_MOUTH_INDICES = list(range(60, 68))
    
    def __init__(
        self,
        mar_threshold: float = 0.75,
        min_yawn_frames: int = 15,
        history_size: int = 30,
        consecutive_yawns_alert: int = 3
    ):
        """
        Initialize the yawn detector.
        
        Args:
            mar_threshold: MAR threshold for yawn detection
            min_yawn_frames: Minimum frames mouth must be open for valid yawn
            history_size: Number of MAR values to keep in history
            consecutive_yawns_alert: Number of yawns in short time to trigger alert
        """
        self.mar_threshold = mar_threshold
        self.min_yawn_frames = min_yawn_frames
        self.history_size = history_size
        self.consecutive_yawns_alert = consecutive_yawns_alert
        
        # Tracking
        self.mar_history: Deque[float] = deque(maxlen=history_size)
        self.yawn_frame_count = 0
        self.total_yawns = 0
        self.is_yawning = False
        self.recent_yawns: Deque[int] = deque(maxlen=100)  # Frame numbers of yawns
        self.frame_number = 0
        
        logger.info(f"Yawn detector initialized (threshold={mar_threshold})")
    
    @staticmethod
    def calculate_mar(mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio.
        
        Uses the inner and outer mouth landmarks to compute a ratio
        that indicates how open the mouth is.
        
        Args:
            mouth_landmarks: Array of mouth landmark coordinates.
                           Expected order: outer (12 points) then inner (8 points)
                           
        Returns:
            Mouth Aspect Ratio value
        """
        # If we have all 20 mouth points (outer + inner)
        if len(mouth_landmarks) >= 20:
            # Use inner mouth for more accurate MAR
            # Inner mouth points: 60-67 in original, or indices 12-19 if passed separately
            # Vertical distances (top to bottom of inner mouth)
            inner_start = 12  # Start of inner mouth in combined array
            
            # Top inner lip to bottom inner lip (vertical opening)
            top_inner = mouth_landmarks[inner_start + 2]  # Point 62
            bottom_inner = mouth_landmarks[inner_start + 6]  # Point 66
            
            # Additional vertical measurements
            top_inner2 = mouth_landmarks[inner_start + 3]  # Point 63
            bottom_inner2 = mouth_landmarks[inner_start + 5]  # Point 65
            
            A = dist.euclidean(top_inner, bottom_inner)
            B = dist.euclidean(top_inner2, bottom_inner2)
            
            # Horizontal distance (mouth width)
            left_corner = mouth_landmarks[inner_start]  # Point 60
            right_corner = mouth_landmarks[inner_start + 4]  # Point 64
            C = dist.euclidean(left_corner, right_corner)
            
        else:
            # Fallback for just outer mouth (12 points)
            # Vertical distances
            A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])  # 50-58
            B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])   # 52-56
            
            # Horizontal distance
            C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])   # 48-54
        
        if C == 0:
            return 0.0
        
        mar = (A + B) / (2.0 * C)
        return mar
    
    def process_mouth(self, mouth_landmarks: np.ndarray) -> YawnMetrics:
        """
        Process mouth landmarks and detect yawning.
        
        Args:
            mouth_landmarks: Array of mouth landmark coordinates
            
        Returns:
            YawnMetrics with current mouth state
        """
        self.frame_number += 1
        
        # Calculate MAR
        mar = self.calculate_mar(mouth_landmarks)
        self.mar_history.append(mar)
        
        # Detect if currently yawning
        currently_open = mar > self.mar_threshold
        
        if currently_open:
            self.yawn_frame_count += 1
        else:
            # Check if we just finished a yawn
            if self.is_yawning and self.yawn_frame_count >= self.min_yawn_frames:
                self.total_yawns += 1
                self.recent_yawns.append(self.frame_number)
                logger.info(f"Yawn detected (total: {self.total_yawns})")
            
            self.yawn_frame_count = 0
        
        self.is_yawning = currently_open and self.yawn_frame_count >= self.min_yawn_frames
        
        return YawnMetrics(
            mar=mar,
            is_yawning=self.is_yawning,
            yawn_duration_frames=self.yawn_frame_count,
            total_yawns=self.total_yawns
        )
    
    def is_fatigue_indicated(self) -> bool:
        """
        Check if yawning pattern indicates fatigue.
        
        Returns:
            True if multiple yawns detected in short time period
        """
        # Count yawns in last ~3 seconds (assuming 30 fps = 90 frames)
        recent_window = 90
        recent_count = sum(
            1 for frame in self.recent_yawns 
            if self.frame_number - frame < recent_window
        )
        
        return recent_count >= self.consecutive_yawns_alert
    
    def get_yawn_frequency(self) -> float:
        """
        Get yawns per minute estimate.
        
        Returns:
            Estimated yawns per minute
        """
        if self.frame_number < 30:  # Need at least 1 second of data
            return 0.0
        
        # Assuming ~30 fps
        minutes = self.frame_number / (30 * 60)
        if minutes == 0:
            return 0.0
        
        return self.total_yawns / minutes
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.mar_history.clear()
        self.yawn_frame_count = 0
        self.total_yawns = 0
        self.is_yawning = False
        self.recent_yawns.clear()
        self.frame_number = 0
        logger.info("Yawn detector reset")
