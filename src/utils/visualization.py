"""
Visualization Utilities

Provides functions for drawing overlays, landmarks, and status information on frames.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# Color constants (BGR format)
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
}


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = COLORS['green'],
    radius: int = 1,
    thickness: int = -1
) -> np.ndarray:
    """
    Draw all facial landmarks on frame.
    
    Args:
        frame: BGR image
        landmarks: (68, 2) array of landmark coordinates
        color: BGR color tuple
        radius: Circle radius
        thickness: -1 for filled circles
        
    Returns:
        Frame with landmarks drawn
    """
    frame = frame.copy()
    for point in landmarks:
        cv2.circle(frame, tuple(point.astype(int)), radius, color, thickness, cv2.LINE_AA)
    return frame


def draw_eye_contours(
    frame: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    color: Tuple[int, int, int] = COLORS['cyan'],
    thickness: int = 1
) -> np.ndarray:
    """
    Draw eye contours on frame.
    
    Args:
        frame: BGR image
        left_eye: (6, 2) array of left eye landmarks
        right_eye: (6, 2) array of right eye landmarks
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with eye contours drawn
    """
    frame = frame.copy()
    
    # Draw convex hulls around eyes
    left_hull = cv2.convexHull(left_eye.astype(np.int32))
    right_hull = cv2.convexHull(right_eye.astype(np.int32))
    
    cv2.drawContours(frame, [left_hull], -1, color, thickness, cv2.LINE_AA)
    cv2.drawContours(frame, [right_hull], -1, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_mouth_contour(
    frame: np.ndarray,
    mouth_landmarks: np.ndarray,
    color: Tuple[int, int, int] = COLORS['magenta'],
    thickness: int = 1
) -> np.ndarray:
    """
    Draw mouth contour on frame.
    
    Args:
        frame: BGR image
        mouth_landmarks: Array of mouth landmarks
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with mouth contour drawn
    """
    frame = frame.copy()
    
    # Draw outer mouth (first 12 points)
    if len(mouth_landmarks) >= 12:
        outer_mouth = mouth_landmarks[:12].astype(np.int32)
        cv2.polylines(frame, [outer_mouth], True, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_drowsiness_overlay(
    frame: np.ndarray,
    is_drowsy: bool,
    drowsiness_score: float,
    blink_count: int,
    ear_value: float,
    is_yawning: bool = False,
    yawn_count: int = 0,
    mar_value: float = 0.0,
    head_pose: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Draw comprehensive drowsiness status overlay on frame.
    
    Args:
        frame: BGR image
        is_drowsy: Whether drowsiness is detected
        drowsiness_score: Drowsiness percentage (0-100)
        blink_count: Total blink count
        ear_value: Current EAR value
        is_yawning: Whether yawning is detected
        yawn_count: Total yawn count
        mar_value: Current MAR value
        head_pose: Optional (pitch, yaw, roll) tuple
        
    Returns:
        Frame with overlay drawn
    """
    frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw drowsiness alert if detected
    if is_drowsy:
        # Red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), COLORS['red'], -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Alert text
        alert_text = "! ! ! DROWSINESS ALERT ! ! !"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(
            frame, alert_text, (text_x, 50),
            cv2.FONT_HERSHEY_DUPLEX, 1.2, COLORS['white'], 2, cv2.LINE_AA
        )
    
    # Draw yawn alert if detected
    if is_yawning:
        cv2.putText(
            frame, "YAWNING DETECTED", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['orange'], 2, cv2.LINE_AA
        )
    
    # Draw status panel (top-right)
    panel_x = width - 200
    panel_y = 10
    line_height = 25
    
    # Background for status panel
    cv2.rectangle(
        frame,
        (panel_x - 10, panel_y - 5),
        (width - 10, panel_y + line_height * 5 + 5),
        (0, 0, 0),
        -1
    )
    cv2.rectangle(
        frame,
        (panel_x - 10, panel_y - 5),
        (width - 10, panel_y + line_height * 5 + 5),
        COLORS['white'],
        1
    )
    
    # Status text
    status_items = [
        f"Score: {drowsiness_score:.1f}%",
        f"EAR: {ear_value:.3f}",
        f"Blinks: {blink_count}",
        f"MAR: {mar_value:.3f}",
        f"Yawns: {yawn_count}",
    ]
    
    for i, text in enumerate(status_items):
        y = panel_y + (i + 1) * line_height
        color = COLORS['red'] if (i == 0 and drowsiness_score > 50) else COLORS['white']
        cv2.putText(
            frame, text, (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    
    # Draw head pose info if available
    if head_pose is not None:
        pitch, yaw, roll = head_pose
        pose_text = f"Pose: P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}"
        cv2.putText(
            frame, pose_text, (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1, cv2.LINE_AA
        )
    
    # Draw drowsiness score bar
    bar_width = 150
    bar_height = 15
    bar_x = 10
    bar_y = height - 50
    
    # Background
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        COLORS['black'],
        -1
    )
    
    # Score fill
    fill_width = int((drowsiness_score / 100) * bar_width)
    if drowsiness_score < 30:
        bar_color = COLORS['green']
    elif drowsiness_score < 60:
        bar_color = COLORS['yellow']
    else:
        bar_color = COLORS['red']
    
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + fill_width, bar_y + bar_height),
        bar_color,
        -1
    )
    
    # Border
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        COLORS['white'],
        1
    )
    
    # Label
    cv2.putText(
        frame, "Drowsiness", (bar_x, bar_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['white'], 1, cv2.LINE_AA
    )
    
    return frame


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30)
) -> np.ndarray:
    """
    Draw FPS counter on frame.
    
    Args:
        frame: BGR image
        fps: Frames per second value
        position: (x, y) position for text
        
    Returns:
        Frame with FPS drawn
    """
    frame = frame.copy()
    cv2.putText(
        frame, f"FPS: {fps:.1f}", position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['green'], 2, cv2.LINE_AA
    )
    return frame


def draw_face_rect(
    frame: np.ndarray,
    face_rect,
    color: Tuple[int, int, int] = COLORS['green'],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw rectangle around detected face.
    
    Args:
        frame: BGR image
        face_rect: dlib rectangle or (x, y, w, h) tuple
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with face rectangle drawn
    """
    frame = frame.copy()
    
    # Handle dlib rectangle
    if hasattr(face_rect, 'left'):
        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()
    else:
        x, y, w, h = face_rect
        x1, y1 = x, y
        x2, y2 = x + w, y + h
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame
