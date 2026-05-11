"""
Configuration and device detection for ESVAL Segmentation Service.
Automatically detects GPU availability and provides fallback to CPU.
"""

import os
import torch
from typing import TypedDict


class DeviceInfo(TypedDict):
    type: str
    name: str
    memory_gb: float | None


def get_device() -> torch.device:
    """
    Detect the best available device for inference.
    Returns CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> DeviceInfo:
    """
    Get detailed information about the inference device.
    Used for health checks and capability reporting.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            type="cuda",
            name=torch.cuda.get_device_name(0),
            memory_gb=round(props.total_memory / (1024**3), 2)
        )
    return DeviceInfo(
        type="cpu",
        name="CPU",
        memory_gb=None
    )


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return torch.cuda.is_available()


# Model configuration
# SEGMENTATION_ENGINE: "mobilesam" or "sam2"
SEGMENTATION_ENGINE = os.environ.get("SEGMENTATION_ENGINE", "mobilesam")

# MobileSAM configuration
MOBILE_SAM_PATH = os.environ.get("MOBILE_SAM_PATH", "models/mobile_sam.pt")
MOBILE_SAM_TYPE = "vit_t"

# SAM 2 configuration
SAM2_MODEL_PATH = os.environ.get("SAM2_MODEL_PATH", "models/sam2_hiera_small.pt")
SAM2_MODEL_SIZE = "sam2_hiera_small"  # sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base_plus, sam2_hiera_large

# Active model path (resolved on init based on engine)
MODEL_PATH = MOBILE_SAM_PATH if SEGMENTATION_ENGINE == "mobilesam" else SAM2_MODEL_PATH
MODEL_TYPE = MOBILE_SAM_TYPE if SEGMENTATION_ENGINE == "mobilesam" else SAM2_MODEL_SIZE

# API configuration
MAX_IMAGE_SIZE = 2048  # Maximum image dimension
MIN_MASK_AREA = 100    # Minimum mask area in pixels

# ── MobileSAM automatic mask generator parameters ───────────────────────────
# Ajustados para imágenes satelitales de propiedades chilenas (zoom alto,
# áreas pequeñas, vegetación de tono verde-oliva poco saturado).
#
# Cambios respecto a valores originales:
#   pred_iou_thresh 0.86 → 0.65  : más máscaras pasan el filtro de IoU
#   stability_score_thresh 0.92 → 0.70 : menos estricto → más cobertura
#   crop_n_layers 0 → 1          : sub-crops detectan features más pequeños
#   crop_overlap_ratio 0.3       : solapamiento entre crops
#   points_per_side 32 → 32      : sin cambio, densidad suficiente
DEFAULT_POINTS_PER_SIDE    = 32
SAM_PRED_IOU_THRESH        = 0.65
SAM_STABILITY_SCORE_THRESH = 0.70
SAM_CROP_N_LAYERS          = 1
SAM_CROP_OVERLAP_RATIO     = 0.3
SAM_MIN_MASK_AREA          = 50   # más pequeño que MIN_MASK_AREA para SAM interno

# ── Rangos HSV para clasificación de respaldo (cuando CLIP falla) ────────────
# Escala OpenCV: H∈[0,179], S∈[0,255], V∈[0,255]
#
# Vegetación en imágenes satelitales chilenas:
#   - Verde brillante (jardines irrigados): H≈60, S>80
#   - Verde oliva / oscuro (árboles desde arriba): H≈25-45, S=30-100
#   - Rango ampliado: H=(20,85) para capturar ambos tonos
#
# Construcción: NO se detecta por rango HSV positivo; es la categoría residual.
# Todo lo que no sea agua, vegetación ni plantación → building (ver _classify_mask_hsv).
ZONE_COLOR_RANGES = {
    "water": {
        # Agua/piscinas: azul brillante, alta saturación
        "hue_range": (90, 130),
        "sat_range": (60, 255),
        "val_range": (80, 255)
    },
    "vegetation": {
        # Verde brillante de jardines: H=40-85, saturación media-alta
        "hue_range": (40, 85),
        "sat_range": (40, 255),
        "val_range": (40, 255)
    },
    "vegetation_olive": {
        # Verde oliva oscuro de árboles/arbustos desde arriba: H=20-45, sat baja
        "hue_range": (20, 45),
        "sat_range": (15, 130),
        "val_range": (30, 160)
    },
    "plantation": {
        # Cultivos en hileras, tono marrón-amarillento
        "hue_range": (10, 30),
        "sat_range": (40, 200),
        "val_range": (60, 200)
    }
}

# Class priority for overlap resolution (higher number = higher priority)
# Water wins over everything (pools should not be classified as vegetation)
CLASS_PRIORITY = {
    "water": 4,        # Highest - pools/water siempre ganan
    "vegetation": 3,   # Second - vegetación prevalece sobre edificio (edificio = residual)
    "plantation": 3,   # Equal to vegetation
    "building": 2,     # Lower - residual de lo que no es agua/vegetación
    "other": 0
}
