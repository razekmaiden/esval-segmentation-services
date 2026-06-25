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
DEFAULT_POINTS_PER_SIDE = 32  # For auto-mask generation
MIN_MASK_AREA = 100  # Minimum mask area in pixels

# Zone classification colors (HSV ranges for detection)
# These correspond to the zone types in the frontend
ZONE_COLOR_RANGES = {
    "water": {
        "hue_range": (90, 130),      # Blue tones
        "sat_range": (50, 255),
        "val_range": (50, 255)
    },
    "vegetation": {
        "hue_range": (35, 85),       # Green tones
        "sat_range": (40, 255),
        "val_range": (40, 255)
    },
    "plantation": {
        "hue_range": (20, 45),       # Brown/yellow-green tones (cultivated)
        "sat_range": (30, 200),
        "val_range": (60, 200)
    }
}

# Class priority for overlap resolution (higher number = higher priority)
# Water wins over everything (pools should not be classified as vegetation)
CLASS_PRIORITY = {
    "water": 4,       # Highest - pools/water always win
    "building": 3,    # Second - clear structures
    "plantation": 2,  # Third - organized crops
    "vegetation": 1,  # Lowest - residual/default category
    "other": 0
}
