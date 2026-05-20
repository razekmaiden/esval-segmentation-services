"""
MobileSAM + SAM 2 + CLIP Segmentation Service.
Handles model loading, inference, mask generation, and semantic classification.

Architecture:
- MobileSAM/SAM 2: Generates precise instance masks
- CLIP: Classifies each mask semantically
- Morphology: Post-processes to remove noisy pixels
"""

import numpy as np
import torch
from PIL import Image
from typing import Any
import cv2
import os

from .config import (
    get_device,
    SEGMENTATION_ENGINE,
    MOBILE_SAM_PATH,
    MOBILE_SAM_TYPE,
    SAM2_MODEL_PATH,
    SAM2_MODEL_SIZE,
    DEFAULT_POINTS_PER_SIDE,
    MIN_MASK_AREA
)
from .clip_classifier import get_clip_classifier, CLIPClassifier
from .morphology import (
    create_class_map_from_masks,
    remove_noise_pixels,
    class_map_to_masks
)


def _build_mobilesam():
    """Build MobileSAM model and predictors."""
    from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

    model = sam_model_registry[MOBILE_SAM_TYPE](checkpoint=MOBILE_SAM_PATH)
    model.to(get_device())
    model.eval()

    predictor = SamPredictor(model)
    auto_generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=DEFAULT_POINTS_PER_SIDE,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=MIN_MASK_AREA
    )

    return model, predictor, auto_generator


def _build_sam2():
    """Build SAM 2 model and predictors."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # SAM 2 model config paths (hydra config paths within the sam2 package)
    # The config files are at sam2/configs/sam2.1/<config>.yaml
    sam2_configs = {
        "sam2_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }

    model_cfg = sam2_configs.get(SAM2_MODEL_SIZE, "configs/sam2.1/sam2.1_hiera_s.yaml")
    checkpoint = SAM2_MODEL_PATH

    # Build SAM 2 model
    model = build_sam2(model_cfg, checkpoint, device=get_device())

    # SAM 2 automatic mask generator
    # Thresholds tuned for satellite/aerial imagery:
    # Lower thresholds vs MobileSAM defaults to achieve comparable coverage (45%+ vs 14%)
    auto_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=DEFAULT_POINTS_PER_SIDE,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.86,
        min_mask_region_area=MIN_MASK_AREA
    )

    return model, None, auto_generator  # predictor created lazily


class SegmentationService:
    """
    Service class for MobileSAM/SAM 2 image segmentation.
    Supports both prompted (point/box) and automatic segmentation.
    """

    def __init__(self, engine: str | None = None):
        self.device = get_device()
        self.model = None
        self.predictor = None
        self.auto_generator = None
        self.clip_classifier: CLIPClassifier | None = None
        self._is_ready = False
        self._use_clip = True  # Toggle for CLIP vs HSV classification
        self._engine = engine if engine is not None else SEGMENTATION_ENGINE

    def load_model(self) -> None:
        """Load the segmentation model and initialize predictors."""
        try:
            if self._engine == "sam2":
                self.model, self.predictor, self.auto_generator = _build_sam2()
                model_name = f"SAM 2 ({SAM2_MODEL_SIZE})"
            else:
                self.model, self.predictor, self.auto_generator = _build_mobilesam()
                model_name = "MobileSAM"

            # Load CLIP classifier for semantic classification
            if self._use_clip:
                try:
                    self.clip_classifier = get_clip_classifier(device=str(self.device))
                    print("CLIP classifier loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load CLIP classifier, falling back to HSV: {e}")
                    self.clip_classifier = None

            self._is_ready = True
            print(f"{model_name} loaded successfully on {self.device}")
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
        if self._engine == "sam2":
            # SAM 2 uses SAM2ImagePredictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if not isinstance(self.predictor, SAM2ImagePredictor):
                self.predictor = SAM2ImagePredictor(self.model)
            self.predictor.set_image(image)

            point_coords = np.array(points) if points else None
            point_labels_arr = np.array(point_labels) if point_labels else None
            box_arr = np.array(box) if box else None

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels_arr,
                box=box_arr,
                multimask_output=True
            )
        else:
            # MobileSAM
            self.predictor.set_image(image)

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
        1. MobileSAM/SAM 2 generates instance masks
        2. CLIP classifies each mask semantically
        3. Morphological post-processing removes noise

        Args:
            image: RGB image as numpy array (H, W, 3)
            apply_morphology: Whether to apply noise removal

        Returns:
            List of mask dictionaries with segmentation data and classification
        """
        # Step 1: Generate masks
        masks = self.auto_generator.generate(image)
        engine_name = "SAM 2" if self._engine == "sam2" else "MobileSAM"
        print(f"{engine_name} generated {len(masks)} masks")

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


def reload_segmentation_service(engine: str) -> SegmentationService:
    """Destroy the current singleton and reload with a new engine."""
    global _service
    _service = None
    _service = SegmentationService(engine=engine)
    _service.load_model()
    return _service
