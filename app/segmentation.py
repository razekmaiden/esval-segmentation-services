"""
MobileSAM Segmentation Service.
Handles model loading, inference, and mask generation.
"""

import numpy as np
import torch
from PIL import Image
from typing import Any
import cv2

from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from .config import (
    get_device, 
    MODEL_PATH, 
    MODEL_TYPE, 
    DEFAULT_POINTS_PER_SIDE,
    MIN_MASK_AREA,
    ZONE_COLOR_RANGES
)


class SegmentationService:
    """
    Service class for MobileSAM-based image segmentation.
    Supports both prompted (point/box) and automatic segmentation.
    """
    
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.predictor = None
        self.auto_generator = None
        self._is_ready = False
        
    def load_model(self) -> None:
        """Load the MobileSAM model and initialize predictors."""
        try:
            self.model = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
            
            self.predictor = SamPredictor(self.model)
            self.auto_generator = SamAutomaticMaskGenerator(
                self.model,
                points_per_side=DEFAULT_POINTS_PER_SIDE,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=MIN_MASK_AREA
            )
            
            self._is_ready = True
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._is_ready = False
            raise
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_ready
    
    def segment_with_prompts(
        self, 
        image: np.ndarray,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        box: list[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment image using point or box prompts.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            points: List of [x, y] coordinates for point prompts
            point_labels: Labels for points (1=foreground, 0=background)
            box: Bounding box [x1, y1, x2, y2] for box prompt
            
        Returns:
            Tuple of (masks array, confidence scores array)
        """
        self.predictor.set_image(image)
        
        # Convert prompts to numpy arrays if provided
        point_coords = np.array(points) if points else None
        point_labels_arr = np.array(point_labels) if point_labels else None
        box_arr = np.array(box) if box else None
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels_arr,
            box=box_arr,
            multimask_output=True
        )
        
        return masks, scores
    
    def auto_segment(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Automatically segment all objects in the image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of mask dictionaries with segmentation data and classification
        """
        masks = self.auto_generator.generate(image)
        
        # Classify each mask based on the underlying image colors
        classified_masks = []
        for mask_data in masks:
            mask = mask_data["segmentation"]
            classification = self._classify_mask(image, mask)
            
            # Only include masks that match our target zones
            if classification in ["water", "vegetation", "plantation"]:
                mask_data["classification"] = classification
                classified_masks.append(mask_data)
        
        return classified_masks
    
    def _classify_mask(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Classify a mask based on the dominant color in the masked region.
        
        Args:
            image: Original RGB image
            mask: Binary mask array
            
        Returns:
            Classification string: 'water', 'vegetation', 'plantation', or 'other'
        """
        # Extract masked region
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return "other"
        
        # Convert to HSV for better color classification
        # Create a small image from the masked pixels for conversion
        mean_color = masked_pixels.mean(axis=0).astype(np.uint8)
        hsv_color = cv2.cvtColor(np.array([[mean_color]]), cv2.COLOR_RGB2HSV)[0][0]
        
        h, s, v = hsv_color
        
        # Check against each zone's color range
        for zone_type, ranges in ZONE_COLOR_RANGES.items():
            h_min, h_max = ranges["hue_range"]
            s_min, s_max = ranges["sat_range"]
            v_min, v_max = ranges["val_range"]
            
            if (h_min <= h <= h_max and 
                s_min <= s <= s_max and 
                v_min <= v <= v_max):
                return zone_type
        
        return "other"


# Global service instance
_service: SegmentationService | None = None


def get_segmentation_service() -> SegmentationService:
    """Get or create the global segmentation service instance."""
    global _service
    if _service is None:
        _service = SegmentationService()
        _service.load_model()
    return _service
