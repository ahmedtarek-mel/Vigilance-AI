"""
Unit tests for the detection modules.

Tests the core algorithms: EAR, MAR, and head pose estimation.
"""

import numpy as np
import pytest
from src.detection.eye_tracker import EyeTracker
from src.detection.yawn_detector import YawnDetector


class TestEyeTracker:
    """Tests for EyeTracker class."""

    def test_calculate_ear_open_eye(self):
        """Test EAR calculation for open eye."""
        # Simulate open eye landmarks (roughly circular shape)
        # Points: left corner, top-left, top-right, right corner, bottom-right, bottom-left
        open_eye = np.array([
            [0, 10],    # Left corner
            [5, 5],     # Top-left
            [15, 5],    # Top-right
            [20, 10],   # Right corner
            [15, 15],   # Bottom-right
            [5, 15],    # Bottom-left
        ])
        
        ear = EyeTracker.calculate_ear(open_eye)
        
        # Open eye should have higher EAR (typically > 0.25)
        assert ear > 0.2, f"Expected EAR > 0.2 for open eye, got {ear}"

    def test_calculate_ear_closed_eye(self):
        """Test EAR calculation for closed eye."""
        # Simulate closed eye landmarks (flattened horizontally)
        closed_eye = np.array([
            [0, 10],    # Left corner
            [5, 9],     # Top-left (very close to bottom)
            [15, 9],    # Top-right (very close to bottom)
            [20, 10],   # Right corner
            [15, 11],   # Bottom-right
            [5, 11],    # Bottom-left
        ])
        
        ear = EyeTracker.calculate_ear(closed_eye)
        
        # Closed eye should have lower EAR (typically < 0.2)
        assert ear < 0.2, f"Expected EAR < 0.2 for closed eye, got {ear}"

    def test_calculate_ear_zero_width(self):
        """Test EAR calculation handles zero width gracefully."""
        # Edge case: zero horizontal distance
        zero_width_eye = np.array([
            [10, 10],
            [10, 5],
            [10, 5],
            [10, 10],  # Same x as left corner = zero width
            [10, 15],
            [10, 15],
        ])
        
        ear = EyeTracker.calculate_ear(zero_width_eye)
        
        # Should return 0.0 to avoid division by zero
        assert ear == 0.0

    def test_process_eyes_tracks_closure(self):
        """Test that eye tracking correctly tracks eye closure."""
        tracker = EyeTracker(ear_threshold=0.25)
        
        # Simulate open eyes
        open_eye = np.array([
            [0, 10], [5, 0], [15, 0], [20, 10], [15, 20], [5, 20]
        ])
        
        result = tracker.process_eyes(open_eye, open_eye)
        assert not result.is_closed
        assert result.closure_duration_frames == 0

    def test_drowsiness_detection(self):
        """Test drowsiness detection after prolonged eye closure."""
        tracker = EyeTracker(ear_threshold=0.25, drowsy_frames_threshold=5)
        
        # Simulate closed eyes for multiple frames
        closed_eye = np.array([
            [0, 10], [5, 9], [15, 9], [20, 10], [15, 11], [5, 11]
        ])
        
        # Process multiple frames with closed eyes
        for _ in range(6):
            tracker.process_eyes(closed_eye, closed_eye)
        
        assert tracker.is_drowsy()

    def test_blink_counting(self):
        """Test that blinks are counted correctly."""
        tracker = EyeTracker(ear_threshold=0.25, drowsy_frames_threshold=10)
        
        open_eye = np.array([
            [0, 10], [5, 0], [15, 0], [20, 10], [15, 20], [5, 20]
        ])
        closed_eye = np.array([
            [0, 10], [5, 9], [15, 9], [20, 10], [15, 11], [5, 11]
        ])
        
        # Simulate a blink: close then open
        tracker.process_eyes(closed_eye, closed_eye)  # Close
        tracker.process_eyes(closed_eye, closed_eye)  # Still closed
        tracker.process_eyes(open_eye, open_eye)      # Open - blink registered
        
        assert tracker.blink_count == 1

    def test_reset(self):
        """Test tracker reset functionality."""
        tracker = EyeTracker()
        tracker.blink_count = 10
        tracker.closed_frame_count = 5
        
        tracker.reset()
        
        assert tracker.blink_count == 0
        assert tracker.closed_frame_count == 0


class TestYawnDetector:
    """Tests for YawnDetector class."""

    def test_calculate_mar_closed_mouth(self):
        """Test MAR calculation for closed mouth."""
        # Simulate closed mouth (20 points: 12 outer + 8 inner)
        closed_mouth = np.zeros((20, 2))
        # Outer mouth (points 0-11)
        closed_mouth[0] = [0, 50]      # Left corner
        closed_mouth[6] = [100, 50]    # Right corner
        # Inner mouth (points 12-19)
        closed_mouth[12] = [20, 50]    # Inner left
        closed_mouth[14] = [50, 48]    # Top
        closed_mouth[16] = [80, 50]    # Inner right
        closed_mouth[18] = [50, 52]    # Bottom (close to top)
        closed_mouth[15] = [60, 48]    # Top2
        closed_mouth[17] = [60, 52]    # Bottom2
        
        mar = YawnDetector.calculate_mar(closed_mouth)
        
        # Closed mouth should have low MAR
        assert mar < 0.5, f"Expected MAR < 0.5 for closed mouth, got {mar}"

    def test_calculate_mar_open_mouth(self):
        """Test MAR calculation for open mouth (yawning)."""
        # Simulate wide open mouth
        open_mouth = np.zeros((20, 2))
        # Outer mouth
        open_mouth[0] = [0, 50]
        open_mouth[6] = [100, 50]
        # Inner mouth with large vertical opening
        open_mouth[12] = [20, 50]      # Inner left
        open_mouth[14] = [50, 20]      # Top (far up)
        open_mouth[16] = [80, 50]      # Inner right
        open_mouth[18] = [50, 80]      # Bottom (far down)
        open_mouth[15] = [60, 20]      # Top2
        open_mouth[17] = [60, 80]      # Bottom2
        
        mar = YawnDetector.calculate_mar(open_mouth)
        
        # Open mouth should have higher MAR
        assert mar > 0.5, f"Expected MAR > 0.5 for open mouth, got {mar}"

    def test_yawn_detection(self):
        """Test yawn detection with sufficient duration."""
        detector = YawnDetector(mar_threshold=0.5, min_yawn_frames=3)
        
        # Closed mouth
        closed = np.zeros((20, 2))
        closed[12] = [20, 50]
        closed[14] = [50, 48]
        closed[16] = [80, 50]
        closed[18] = [50, 52]
        closed[15] = [60, 48]
        closed[17] = [60, 52]
        
        # Open mouth
        open_m = np.zeros((20, 2))
        open_m[12] = [20, 50]
        open_m[14] = [50, 20]
        open_m[16] = [80, 50]
        open_m[18] = [50, 80]
        open_m[15] = [60, 20]
        open_m[17] = [60, 80]
        
        # Yawn sequence: open for 4 frames then close
        for _ in range(4):
            detector.process_mouth(open_m)
        detector.process_mouth(closed)  # Close - yawn should be registered
        
        assert detector.total_yawns == 1

    def test_reset(self):
        """Test detector reset functionality."""
        detector = YawnDetector()
        detector.total_yawns = 5
        detector.yawn_frame_count = 10
        
        detector.reset()
        
        assert detector.total_yawns == 0
        assert detector.yawn_frame_count == 0


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test that default config has sensible values."""
        from src.core.drowsiness_detector import Config
        
        config = Config()
        
        assert config.ear_threshold == 0.25
        assert config.mar_threshold == 0.75
        assert config.camera_fps == 30
        assert config.sound_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
