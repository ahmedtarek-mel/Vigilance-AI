"""
Image Processing Utilities

Provides functions for preprocessing frames to improve detection accuracy.
"""

import cv2
import numpy as np


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Apply gamma correction to brighten or darken an image.
    
    Gamma < 1: Brightens the image
    Gamma > 1: Darkens the image
    
    Args:
        image: BGR image
        gamma: Gamma value for correction
        
    Returns:
        Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in range(256)
    ]).astype(np.uint8)
    
    return cv2.LUT(image, table)


def histogram_equalization(image: np.ndarray, color: bool = False) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        image: BGR or grayscale image
        color: If True, apply CLAHE to color image (LAB color space)
        
    Returns:
        Equalized image
    """
    if len(image.shape) == 2:
        # Grayscale
        return cv2.equalizeHist(image)
    
    if not color:
        # Convert to grayscale and equalize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    
    # Color image - use CLAHE on L channel in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(
    frame: np.ndarray,
    target_height: int = 480,
    apply_histogram_eq: bool = True,
    apply_gamma: bool = False,
    gamma_value: float = 1.5
) -> tuple[np.ndarray, float]:
    """
    Preprocess frame for face detection.
    
    Args:
        frame: BGR image from camera
        target_height: Target height for resizing
        apply_histogram_eq: Whether to apply histogram equalization
        apply_gamma: Whether to apply gamma correction
        gamma_value: Gamma value if applying gamma correction
        
    Returns:
        Tuple of (processed_frame, resize_ratio)
    """
    # Calculate resize ratio
    height, width = frame.shape[:2]
    resize_ratio = height / target_height
    
    # Resize if needed
    if resize_ratio != 1.0:
        new_width = int(width / resize_ratio)
        frame = cv2.resize(
            frame,
            (new_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Apply enhancements
    if apply_gamma:
        frame = gamma_correction(frame, gamma_value)
    
    if apply_histogram_eq:
        # Keep color for display, return grayscale for detection
        pass  # Detection methods handle their own grayscale conversion
    
    return frame, resize_ratio


def denoise_frame(frame: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply fast denoising to frame.
    
    Args:
        frame: BGR image
        strength: Denoising strength (higher = more smoothing)
        
    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: int = 0
) -> np.ndarray:
    """
    Adjust brightness and contrast of an image.
    
    Args:
        image: BGR image
        brightness: Brightness adjustment (-127 to 127)
        contrast: Contrast adjustment (-127 to 127)
        
    Returns:
        Adjusted image
    """
    brightness = int((brightness - 0) * (255 - (-255)) / (127 - (-127)) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (127 - (-127)) + (-127))

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max_val = 255
        else:
            shadow = 0
            max_val = 255 + brightness
        
        alpha = (max_val - shadow) / 255
        gamma = shadow
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

    if contrast != 0:
        alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma = 127 * (1 - alpha)
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

    return image
