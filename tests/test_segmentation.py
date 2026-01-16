"""
Tests for the segmentation service.
"""

import pytest
import numpy as np
from PIL import Image
import io


def create_test_image(width: int = 256, height: int = 256) -> bytes:
    """Create a simple test image with colored regions."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Blue region (water)
    img[50:100, 50:100] = [0, 100, 200]
    
    # Green region (vegetation)
    img[100:150, 100:150] = [34, 139, 34]
    
    # Brown/yellow region (plantation)
    img[150:200, 50:100] = [160, 140, 80]
    
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestUtils:
    """Test utility functions."""
    
    def test_load_image_from_bytes(self):
        from app.utils import load_image_from_bytes
        
        img_bytes = create_test_image()
        result = load_image_from_bytes(img_bytes)
        
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
    
    def test_resize_image_if_needed(self):
        from app.utils import resize_image_if_needed
        
        # Small image - no resize
        small_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result, scale = resize_image_if_needed(small_img, 2048)
        assert scale == 1.0
        assert result.shape == small_img.shape
        
        # Large image - should resize
        large_img = np.zeros((4000, 3000, 3), dtype=np.uint8)
        result, scale = resize_image_if_needed(large_img, 2048)
        assert scale < 1.0
        assert max(result.shape[:2]) <= 2048
    
    def test_mask_to_polygon(self):
        from app.utils import mask_to_polygon
        
        # Create a simple rectangular mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        
        polygons = mask_to_polygon(mask)
        
        assert len(polygons) > 0
        assert len(polygons[0]) >= 4  # At least 3 points + closing
    
    def test_pixel_coords_to_geo(self):
        from app.utils import pixel_coords_to_geo
        
        bounds = {
            "north": -33.0,
            "south": -33.1,
            "east": -71.5,
            "west": -71.6
        }
        image_size = (1000, 1000)
        
        # Top-left corner should be northwest
        coords = pixel_coords_to_geo([[0, 0]], bounds, image_size)
        assert coords[0][0] == pytest.approx(-71.6)  # west
        assert coords[0][1] == pytest.approx(-33.0)  # north
        
        # Bottom-right should be southeast
        coords = pixel_coords_to_geo([[1000, 1000]], bounds, image_size)
        assert coords[0][0] == pytest.approx(-71.5)  # east
        assert coords[0][1] == pytest.approx(-33.1)  # south


class TestConfig:
    """Test configuration functions."""
    
    def test_get_device(self):
        from app.config import get_device
        import torch
        
        device = get_device()
        assert device.type in ["cuda", "cpu"]
    
    def test_get_device_info(self):
        from app.config import get_device_info
        
        info = get_device_info()
        assert "type" in info
        assert "name" in info
        assert info["type"] in ["cuda", "cpu"]
