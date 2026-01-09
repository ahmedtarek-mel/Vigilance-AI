"""
Video Stream Module

Handles camera capture with threading for improved performance.
"""

import logging
import time
from threading import Thread
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Threaded video capture class for better performance.
    
    Uses a separate thread to continuously read frames from the camera,
    reducing latency in the main processing loop.
    
    Attributes:
        device_id: Camera device index
        width: Frame width
        height: Frame height
        fps: Target FPS
    """
    
    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Initialize video stream.
        
        Args:
            device_id: Camera device index (0 = default webcam)
            width: Desired frame width
            height: Desired frame height
            fps: Target frames per second
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.grabbed: bool = False
        self.stopped: bool = True
        self.thread: Optional[Thread] = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = 0.0
        self.actual_fps = 0.0
    
    def start(self) -> 'VideoStream':
        """
        Start the video stream.
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting video stream (device={self.device_id})")
        
        # Initialize capture with DirectShow backend on Windows
        # This avoids MSMF errors like -1072875819
        import platform
        if platform.system() == 'Windows':
            self.capture = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(self.device_id)
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.device_id}")
        
        # Set camera properties
        # Note: Commented out because enforcing resolution/FPS breaks some DSHOW drivers
        # resulting in black frames. We'll use the camera's defaults.
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read actual values (camera may not support requested values)
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Warm up camera (skip first few frames)
        for _ in range(10):
            self.capture.read()
        
        # Read first frame
        self.grabbed, self.frame = self.capture.read()
        
        if not self.grabbed:
            raise RuntimeError("Could not read from camera")
        
        # Start background thread
        self.stopped = False
        self.start_time = time.time()
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        
        logger.info("Video stream started")
        return self
    
    def _update(self) -> None:
        """Background thread that continuously reads frames."""
        while not self.stopped:
            if self.capture is None:
                break
            
            self.grabbed, self.frame = self.capture.read()
            
            if self.grabbed:
                self.frame_count += 1
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the most recent frame.
        
        Returns:
            Tuple of (success, frame)
        """
        return self.grabbed, self.frame.copy() if self.frame is not None else None
    
    def get_fps(self) -> float:
        """
        Get actual FPS based on frames read.
        
        Returns:
            Actual frames per second
        """
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.actual_fps = self.frame_count / elapsed
        return self.actual_fps
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get actual frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.frame is not None:
            return self.frame.shape[1], self.frame.shape[0]
        return self.width, self.height
    
    def stop(self) -> None:
        """Stop the video stream and release resources."""
        logger.info("Stopping video stream...")
        
        self.stopped = True
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        
        logger.info("Video stream stopped")
    
    def __enter__(self) -> 'VideoStream':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


class VideoWriter:
    """
    Video writer for recording output.
    
    Attributes:
        output_path: Path to output video file
        fps: Frames per second for output
    """
    
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video file
            fps: Frames per second
            codec: FourCC codec code
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_size: Optional[Tuple[int, int]] = None
        self.frame_count = 0
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: BGR image to write
        """
        if self.writer is None:
            # Initialize on first frame
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )
            
            logger.info(f"Video writer initialized: {self.output_path}")
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self) -> None:
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"Video saved: {self.output_path} ({self.frame_count} frames)")
    
    def __enter__(self) -> 'VideoWriter':
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.release()
