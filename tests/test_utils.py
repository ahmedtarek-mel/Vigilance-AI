"""
Unit tests for utility modules.

Tests image processing and visualization functions.
"""

import numpy as np
import pytest
from src.utils.image_processing import gamma_correction, histogram_equalization


class TestImageProcessing:
    """Tests for image processing utilities."""

    def test_gamma_correction_brightens(self):
        """Test that gamma > 1 brightens the image (inverse gamma correction)."""
        # Create a dark gray image
        dark_image = np.full((100, 100, 3), 50, dtype=np.uint8)
        
        # The function uses inv_gamma = 1/gamma, so gamma > 1 brightens
        brightened = gamma_correction(dark_image, gamma=2.0)
        
        # Average should be higher after brightening
        assert brightened.mean() > dark_image.mean()

    def test_gamma_correction_darkens(self):
        """Test that gamma < 1 darkens the image (inverse gamma correction)."""
        # Create a light gray image
        light_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        
        # The function uses inv_gamma = 1/gamma, so gamma < 1 darkens
        darkened = gamma_correction(light_image, gamma=0.5)
        
        # Average should be lower after darkening
        assert darkened.mean() < light_image.mean()

    def test_gamma_correction_preserves_shape(self):
        """Test that gamma correction preserves image shape."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        result = gamma_correction(image, gamma=1.5)
        
        assert result.shape == image.shape

    def test_histogram_equalization_grayscale(self):
        """Test histogram equalization on grayscale image."""
        # Create a low contrast grayscale image
        gray_image = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        
        equalized = histogram_equalization(gray_image, color=False)
        
        # Result should have better contrast (wider range)
        assert equalized.std() >= gray_image.std()

    def test_histogram_equalization_color(self):
        """Test histogram equalization on color image."""
        # Create a color image
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        equalized = histogram_equalization(color_image, color=True)
        
        # Should return same shape color image
        assert equalized.shape == color_image.shape


class TestVisualization:
    """Tests for visualization utilities."""

    def test_draw_landmarks_returns_image(self):
        """Test that draw_landmarks returns an image."""
        from src.utils.visualization import draw_landmarks
        
        # Create a test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = np.random.randint(0, 480, (68, 2))
        
        result = draw_landmarks(image, landmarks)
        
        assert result.shape == image.shape
        assert result is not image  # Should be a copy

    def test_draw_eye_contours(self):
        """Test that draw_eye_contours draws on image."""
        from src.utils.visualization import draw_eye_contours
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        left_eye = np.array([[100, 100], [110, 90], [130, 90], [140, 100], [130, 110], [110, 110]])
        right_eye = left_eye + [200, 0]
        
        result = draw_eye_contours(image, left_eye, right_eye)
        
        # Image should have some non-zero pixels (drawn contours)
        assert result.sum() > 0

    def test_draw_fps(self):
        """Test FPS drawing."""
        from src.utils.visualization import draw_fps
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = draw_fps(image, 30.5)
        
        # Should have drawn text (non-zero pixels)
        assert result.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
