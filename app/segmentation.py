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
    MIN_MASK_AREA,
    SAM_PRED_IOU_THRESH,
    SAM_STABILITY_SCORE_THRESH,
    SAM_CROP_N_LAYERS,
    SAM_CROP_OVERLAP_RATIO,
    SAM_MIN_MASK_AREA,
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
    # Parámetros ajustados para imágenes satelitales de propiedades chilenas.
    # pred_iou_thresh y stability_score_thresh bajados para generar más máscaras.
    # crop_n_layers=1 detecta features pequeños que el grid global pierde.
    auto_generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=DEFAULT_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        crop_n_layers=SAM_CROP_N_LAYERS,
        crop_overlap_ratio=SAM_CROP_OVERLAP_RATIO,
        min_mask_region_area=SAM_MIN_MASK_AREA,
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
    auto_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=DEFAULT_POINTS_PER_SIDE,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=MIN_MASK_AREA
    )

    return model, None, auto_generator  # predictor created lazily


class SegmentationService:
    """
    Service class for MobileSAM/SAM 2 image segmentation.
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
        self._engine = SEGMENTATION_ENGINE

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
            classified_masks = self._apply_morphology(classified_masks, image.shape[:2], image)
            print(f"After morphology: {len(classified_masks)} clean masks")

        return classified_masks
    
    def _apply_morphology(
        self,
        masks: list[dict[str, Any]],
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Aplica post-procesamiento morfológico para limpiar píxeles ruidosos.

        Pipeline:
        1. Crear class map unificado (máscaras grandes primero, prioridad vegetal > building)
        2. Limpieza morfológica (erosión/dilatación con kernel pequeño)
        3. Rellenar píxeles no clasificados (0) por vecino más cercano
        4. Corrección pixel-a-pixel: reclasificar píxeles de edificio que son vegetación
        5. Convertir de vuelta a máscaras por clase
        """
        # 1. Construir class map con sistema de prioridad correcto
        class_map = create_class_map_from_masks(masks, image_shape)

        # 2. Limpieza morfológica ligera
        cleaned_map = remove_noise_pixels(
            class_map,
            min_region_size=30,
            kernel_size=3
        )

        # 3. Rellenar píxeles no clasificados (0) que quedaron tras la erosión.
        #    Se usa distance_transform_edt para asignar a cada píxel 0 el valor
        #    del píxel clasificado más cercano (nearest-neighbor fill).
        unclassified = cleaned_map == 0
        if unclassified.any() and (cleaned_map > 0).any():
            from scipy.ndimage import distance_transform_edt
            _, indices = distance_transform_edt(unclassified, return_indices=True)
            cleaned_map[unclassified] = cleaned_map[indices[0][unclassified],
                                                     indices[1][unclassified]]

        # 4. Corrección pixel-a-pixel de vegetación.
        #    SAM clasifica por máscara (un segmento = una clase). Si un segmento
        #    mezcla edificio y vegetación, se etiqueta como edificio (residual).
        #    Este paso reclasifica individualmente los píxeles de edificio que
        #    tienen firma espectral de vegetación (ExG alto + relación verde alta).
        if image is not None:
            BUILDING_ID   = 4  # según create_class_map_from_masks
            VEGETATION_ID = 2
            building_mask = cleaned_map == BUILDING_ID
            if building_mask.any():
                r = image[:, :, 0].astype(np.float32)
                g = image[:, :, 1].astype(np.float32)
                b = image[:, :, 2].astype(np.float32)
                exg = 2 * g - r - b
                denom = r + g + b
                green_ratio = np.where(denom > 0, g / denom, 0.0)

                # Criterio conservador: ExG fuerte + dominancia verde clara
                is_veg = (exg > 5) & (green_ratio > 0.33)
                # Criterio alternativo con umbral más suave pero señal verde neta
                is_veg |= (exg > 3) & (green_ratio > 0.34)
                # Criterio amplio: cualquier píxel con verde dominante sobre rojo
                is_veg |= (exg > 1) & (green_ratio > 0.35)

                # Solo reclasificar píxeles que estaban marcados como edificio
                reclassify = building_mask & is_veg
                cleaned_map[reclassify] = VEGETATION_ID
                n_reclassified = int(reclassify.sum())
                if n_reclassified > 0:
                    print(f"Pixel correction: {n_reclassified} building pixels reclassified as vegetation")

        # 5. Convertir class map → lista de máscaras por clase
        cleaned_masks = class_map_to_masks(cleaned_map)

        return cleaned_masks
    
    def _classify_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> tuple[str, float]:
        """
        Clasifica una máscara usando CLIP (preferido) con fallback a HSV.

        CLIP es más semántico pero puede devolver "other" cuando la confianza
        es baja (imagen muy pequeña, colores ambiguos).  En ese caso se usa HSV
        como clasificador de respaldo en lugar de descartar la máscara.
        """
        if self.clip_classifier is not None:
            cls, conf = self.clip_classifier.classify_mask(image, mask)
            # Si CLIP produce una clase válida, usarla directamente
            if cls != "other":
                return cls, conf
            # CLIP no pudo clasificar → intentar con HSV antes de descartar

        return self._classify_mask_hsv(image, mask)
    
    def _classify_mask_hsv(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> tuple[str, float]:
        """
        Clasificación HSV robusta para imágenes satelitales chilenas.

        Estrategia:
          1. Vegetación: detección por ExG (Excess Green Index) y ratio de verde.
             Cubre tanto verde brillante (jardines) como verde oliva oscuro
             (árboles desde arriba, H≈25-45 en OpenCV).
          2. Agua/piscina: azul dominante con saturación y brillo altos.
          3. Construcción: categoría residual para todo lo que no es vegetación
             ni agua. Abarca techos grises, naranjas, marrones, metal, concreto.

        No se clasifica nada como "other": un pixel dentro del predio que no es
        vegetación ni agua muy probablemente es construcción u otra superficie.
        """
        masked_pixels = image[mask]
        if len(masked_pixels) < 10:
            return "other", 0.0

        avg = masked_pixels.mean(axis=0)
        r, g, b = float(avg[0]), float(avg[1]), float(avg[2])
        total = r + g + b + 1e-6

        # ── Índices colorimétricos ──────────────────────────────────────────
        green_ratio = g / total
        blue_ratio  = b / total
        exg = 2 * g - r - b            # Excess Green Index

        # HSV promedio de los píxeles de la máscara
        bgr_pixels = masked_pixels[:, ::-1].copy().reshape(-1, 1, 3).astype(np.uint8)
        hsv_pixels = cv2.cvtColor(bgr_pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        h_mean = hsv_pixels[:, 0].mean()   # OpenCV H: 0-179
        s_mean = hsv_pixels[:, 1].mean()   # 0-255
        v_mean = hsv_pixels[:, 2].mean()   # 0-255
        s_std  = hsv_pixels[:, 1].std()

        # ── 1. VEGETACIÓN ──────────────────────────────────────────────────
        # Criterio A: ExG alto → señal de clorofila clara (jardines irrigados)
        if exg > 5 and green_ratio > 0.33:
            n = len(masked_pixels)
            if n > 50:
                color_std = masked_pixels.std(axis=0).mean()
                if color_std < 18 and s_mean > 45:
                    return "plantation", 0.7  # vegetación uniforme → plantación
            return "vegetation", 0.7

        # Criterio B: ExG moderado con dominancia verde y brillo suficiente
        if exg > 3 and green_ratio > 0.34 and g > r and v_mean > 50:
            return "vegetation", 0.6

        # Criterio C: Verde oliva / oscuro (H=20-55 en OpenCV, s baja)
        # Captura árboles desde arriba cuyo tono es marrón-verde (H≈25-45)
        if 20 <= h_mean <= 55 and s_mean > 15 and g > r and green_ratio > 0.32:
            return "vegetation", 0.6

        # Criterio D: Verde dominante en RGB sin señal HSV fuerte
        if g > r and g > b and green_ratio > 0.34:
            if s_mean > 12 or (g > 55 and exg > 2):
                return "vegetation", 0.55

        # Criterio E: Verde pálido / seco, brillo alto (pasto seco de verano)
        if exg > 1 and green_ratio > 0.35 and v_mean > 85 and g > r:
            return "vegetation", 0.5

        # ── 2. AGUA / PISCINA ──────────────────────────────────────────────
        # Piscinas: azul claro bien saturado, NO demasiado oscuro
        is_water = False
        if 90 <= h_mean <= 130 and s_mean > 70 and v_mean > 90:
            is_water = True
        if blue_ratio > 0.42 and b > r * 1.3 and b > g * 1.2 and s_mean > 50 and v_mean > 80:
            is_water = True
        # Rechazar agua oscura (sombras/techos oscuros)
        if is_water and v_mean < 75:
            is_water = False
        # Rechazar agua no uniforme (las piscinas tienen color muy uniforme)
        if is_water and s_std > 45:
            is_water = False
        if is_water:
            return "water", 0.8

        # ── 3. CONSTRUCCIÓN (residual) ─────────────────────────────────────
        # Todo lo que no es vegetación ni agua se clasifica como construcción.
        # Esto abarca: techos grises, rojos/naranjas, metálicos, pavimento,
        # concreto, asfalto. Es la categoría dominante en zonas urbanas.
        return "building", 0.5


# Global service instance
_service: SegmentationService | None = None


def get_segmentation_service() -> SegmentationService:
    """Get or create the global segmentation service instance."""
    global _service
    if _service is None:
        _service = SegmentationService()
        _service.load_model()
    return _service
