"""
MobileSAM + CLIP Segmentation Service.
Handles model loading, inference, mask generation, and semantic classification.

Architecture:
- MobileSAM: Generates precise instance masks
- CLIP: Classifies each mask semantically
- Morphology: Post-processes to remove noisy pixels
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
    MIN_MASK_AREA
)
from .clip_classifier import get_clip_classifier, CLIPClassifier
from .morphology import (
    create_class_map_from_masks,
    remove_noise_pixels,
    class_map_to_masks
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
        self.clip_classifier: CLIPClassifier | None = None
        self._is_ready = False
        self._use_clip = True  # Toggle for CLIP vs HSV classification
        
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
            
            # Load CLIP classifier for semantic classification
            if self._use_clip:
                try:
                    self.clip_classifier = get_clip_classifier(device=str(self.device))
                    print("CLIP classifier loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load CLIP classifier, falling back to HSV: {e}")
                    self.clip_classifier = None
            
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
    
    def auto_segment(
        self, 
        image: np.ndarray,
        apply_morphology: bool = True
    ) -> list[dict[str, Any]]:
        """
        Automatically segment all objects in the image.
        
        Pipeline:
        1. MobileSAM generates instance masks
        2. CLIP classifies each mask semantically
        3. Morphological post-processing removes noise
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            apply_morphology: Whether to apply noise removal
            
        Returns:
            List of mask dictionaries with segmentation data and classification
        """
        # Step 1: Generate masks with MobileSAM
        masks = self.auto_generator.generate(image)
        print(f"MobileSAM generated {len(masks)} masks")
        
        # Step 2: Classify each mask using CLIP or HSV
        classified_masks = []
        for mask_data in masks:
            mask = mask_data["segmentation"]
            classification, confidence = self._classify_mask(image, mask)
            
            # Include masks that match our target zones (now including building)
            if classification in ["water", "vegetation", "plantation", "building"]:
                mask_data["classification"] = classification
                mask_data["confidence"] = confidence
                classified_masks.append(mask_data)
        
        print(f"Classified {len(classified_masks)} masks with target classes")
        
        # Step 3: Apply morphological post-processing
        if apply_morphology and len(classified_masks) > 0:
            classified_masks = self._apply_morphology(classified_masks, image.shape[:2])
            print(f"After morphology: {len(classified_masks)} clean masks")
        
        return classified_masks
    
    def _apply_morphology(
        self, 
        masks: list[dict[str, Any]], 
        image_shape: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """
        Apply morphological post-processing to remove noisy pixels.
        
        Args:
            masks: List of classified mask dictionaries
            image_shape: (height, width) of the image
            
        Returns:
            Cleaned mask dictionaries
        """
        # Create unified class map from all masks
        class_map = create_class_map_from_masks(masks, image_shape)
        
        # Remove noise
        cleaned_map = remove_noise_pixels(
            class_map, 
            min_region_size=50,
            kernel_size=5
        )
        
        # Convert back to mask format
        cleaned_masks = class_map_to_masks(cleaned_map)
        
        return cleaned_masks
    
    def _classify_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> tuple[str, float]:
        """
        Classify a mask using CLIP (preferred) or HSV fallback.
        
        Args:
            image: Original RGB image
            mask: Binary mask array
            
        Returns:
            Tuple of (classification string, confidence score)
        """
        # Use CLIP if available
        if self.clip_classifier is not None:
            return self.clip_classifier.classify_mask(image, mask)
        
        # Fallback to HSV classification
        return self._classify_mask_hsv(image, mask)
    
    def _classify_mask_hsv(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> tuple[str, float]:
        """
        Fallback HSV-based classification.
        
        Args:
            image: Original RGB image
            mask: Binary mask array
            
        Returns:
            Tuple of (classification string, confidence score)
        """
        from .config import ZONE_COLOR_RANGES
        
        # Extract masked region
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return "other", 0.0
        
        # Convert to HSV for color classification
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
                # HSV doesn't provide real confidence, use 0.5 as default
                return zone_type, 0.5
        
        return "other", 0.0


# Global service instance
_service: SegmentationService | None = None


def get_segmentation_service() -> SegmentationService:
    """Get or create the global segmentation service instance."""
    global _service
    if _service is None:
        _service = SegmentationService()
        _service.load_model()
    return _service
