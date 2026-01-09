"""
Eye Tracking Module

Implements Eye Aspect Ratio (EAR) calculation for blink and drowsiness detection.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)


@dataclass
class EyeMetrics:
    """Container for eye-related metrics."""
    left_ear: float
    right_ear: float
    avg_ear: float
    is_closed: bool
    closure_duration_frames: int


class EyeTracker:
    """
    Eye tracker using Eye Aspect Ratio (EAR) algorithm.
    
    The EAR formula:
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    Where p1-p6 are the 6 landmark points around each eye.
    
    Attributes:
        ear_threshold: EAR value below which eyes are considered closed
        history_size: Number of frames to keep in EAR history
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.25,
        history_size: int = 30,
        drowsy_frames_threshold: int = 45
    ):
        """
        Initialize the eye tracker.
        
        Args:
            ear_threshold: EAR threshold for closed eyes detection
            history_size: Number of EAR values to keep in history
            drowsy_frames_threshold: Consecutive closed frames to trigger drowsiness
        """
        self.ear_threshold = ear_threshold
        self.history_size = history_size
        self.drowsy_frames_threshold = drowsy_frames_threshold
        
        # History tracking
        self.ear_history: Deque[float] = deque(maxlen=history_size)
        self.closed_frame_count = 0
        self.blink_count = 0
        self.is_blinking = False
        
        logger.info(f"Eye tracker initialized (threshold={ear_threshold})")
    
    @staticmethod
    def calculate_ear(eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio for a single eye.
        
        Args:
            eye_landmarks: (6, 2) array of eye landmark coordinates
            
        Returns:
            Eye Aspect Ratio value
        """
        # Compute euclidean distances between vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def process_eyes(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> EyeMetrics:
        """
        Process eye landmarks and return metrics.
        
        Args:
            left_eye: (6, 2) array of left eye landmarks
            right_eye: (6, 2) array of right eye landmarks
            
        Returns:
            EyeMetrics with current eye state
        """
        # Calculate EAR for each eye
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to history
        self.ear_history.append(avg_ear)
        
        # Determine if eyes are closed
        is_closed = avg_ear < self.ear_threshold
        
        # Track eye closure
        if is_closed:
            self.closed_frame_count += 1
        else:
            # Check for blink (was closed, now open)
            if self.is_blinking and self.closed_frame_count > 0:
                # Only count as blink if it was a short closure
                if self.closed_frame_count < self.drowsy_frames_threshold:
                    self.blink_count += 1
                    logger.debug(f"Blink detected (total: {self.blink_count})")
            
            self.closed_frame_count = 0
        
        self.is_blinking = is_closed
        
        return EyeMetrics(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            is_closed=is_closed,
            closure_duration_frames=self.closed_frame_count
        )
    
    def is_drowsy(self) -> bool:
        """
        Check if the user is drowsy based on eye closure duration.
        
        Returns:
            True if eyes have been closed for too long
        """
        return self.closed_frame_count >= self.drowsy_frames_threshold
    
    def get_drowsiness_score(self) -> float:
        """
        Get drowsiness score from 0-100.
        
        Returns:
            Drowsiness percentage based on recent EAR values
        """
        if not self.ear_history:
            return 0.0
        
        # Calculate how often eyes were below threshold recently
        threshold_crossings = sum(
            1 for ear in self.ear_history if ear < self.ear_threshold
        )
        
        # Also factor in current closure duration
        closure_factor = min(self.closed_frame_count / self.drowsy_frames_threshold, 1.0)
        
        # Combined score
        history_score = (threshold_crossings / len(self.ear_history)) * 50
        closure_score = closure_factor * 50
        
        return min(history_score + closure_score, 100.0)
    
    def get_ear_trend(self) -> str:
        """
        Analyze EAR trend.
        
        Returns:
            'stable', 'declining', or 'recovering'
        """
        if len(self.ear_history) < 10:
            return 'stable'
        
        recent = list(self.ear_history)[-10:]
        older = list(self.ear_history)[:10]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff < -0.02:
            return 'declining'
        elif diff > 0.02:
            return 'recovering'
        return 'stable'
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.ear_history.clear()
        self.closed_frame_count = 0
        self.blink_count = 0
        self.is_blinking = False
        logger.info("Eye tracker reset")
