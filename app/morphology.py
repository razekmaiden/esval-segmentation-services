"""
Morphological post-processing for semantic segmentation cleanup.

Removes noisy isolated pixels and small regions by reassigning them
to the dominant class in their neighborhood.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Optional


def remove_noise_pixels(
    class_map: np.ndarray, 
    min_region_size: int = 50,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Remove noisy pixels from a class map using morphological operations.
    
    Pipeline:
    1. Morphological opening/closing per class to smooth boundaries
    2. Remove small connected components by reassigning to neighbor class
    3. Mode filter for remaining isolated pixels
    
    Args:
        class_map: 2D array where each pixel has a class ID (0=background, 1-4=classes)
        min_region_size: Minimum connected component size in pixels to keep
        kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned class map with same shape
    """
    cleaned = class_map.copy()
    unique_classes = np.unique(class_map)
    
    # Step 1: Morphological opening/closing per class
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
            
        binary_mask = (class_map == class_id).astype(np.uint8)
        
        # Opening removes small noise (erode then dilate)
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # Closing fills small holes (dilate then erode)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Update only this class's pixels
        cleaned[class_map == class_id] = 0  # Clear old positions
        cleaned[closed == 1] = class_id  # Set cleaned positions
    
    # Step 2: Remove small connected components
    cleaned = _remove_small_components(cleaned, min_region_size)
    
    # Step 3: Mode filter for remaining isolated pixels
    cleaned = _apply_mode_filter(cleaned, filter_size=3)
    
    return cleaned


def _remove_small_components(
    class_map: np.ndarray, 
    min_size: int
) -> np.ndarray:
    """
    Remove connected components smaller than min_size pixels.
    Small regions are reassigned to the most common neighboring class.
    """
    cleaned = class_map.copy()
    unique_classes = np.unique(class_map)
    
    for class_id in unique_classes:
        if class_id == 0:
            continue
            
        binary_mask = (cleaned == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        
        for label in range(1, num_labels):  # Skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            
            if area < min_size:
                # Get region mask
                region_mask = labels == label
                
                # Find dominant neighbor class
                new_class = _get_dominant_neighbor(cleaned, region_mask, class_id)
                
                # Reassign pixels
                cleaned[region_mask] = new_class
    
    return cleaned


def _get_dominant_neighbor(
    class_map: np.ndarray, 
    region_mask: np.ndarray, 
    current_class: int
) -> int:
    """
    Get the most common class in the neighborhood of a region.
    
    Args:
        class_map: The full class map
        region_mask: Boolean mask of the region to check neighbors for
        current_class: Current class of the region (to exclude from neighbors)
        
    Returns:
        Class ID of the dominant neighbor, or current_class if no valid neighbors
    """
    # Dilate the region to get its neighbors
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(region_mask.astype(np.uint8), kernel)
    
    # Neighbor mask = dilated minus original region
    neighbor_mask = dilated.astype(bool) & ~region_mask
    
    if not neighbor_mask.any():
        return current_class
    
    # Get neighbor class values (excluding current class and background)
    neighbor_classes = class_map[neighbor_mask]
    neighbor_classes = neighbor_classes[
        (neighbor_classes != current_class) & (neighbor_classes != 0)
    ]
    
    if len(neighbor_classes) == 0:
        return current_class
    
    # Return mode (most common class)
    values, counts = np.unique(neighbor_classes, return_counts=True)
    return int(values[np.argmax(counts)])


def _apply_mode_filter(class_map: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """
    Apply mode filter - each pixel takes the most common value in its neighborhood.
    
    This smooths remaining isolated pixels that don't form connected components.
    """
    def mode_func(values):
        """Calculate mode of a flattened window."""
        values = values.astype(int)
        # Remove any potential NaN from padding
        valid_values = values[values >= 0]
        if len(valid_values) == 0:
            return 0
        counts = np.bincount(valid_values)
        return np.argmax(counts)
    
    # Apply generic filter with mode function
    filtered = ndimage.generic_filter(
        class_map.astype(float), 
        mode_func, 
        size=filter_size, 
        mode='nearest'
    )
    
    return filtered.astype(class_map.dtype)


def create_class_map_from_masks(
    masks: list[dict], 
    image_shape: tuple[int, int],
    class_to_id: Optional[dict] = None
) -> np.ndarray:
    """
    Create a unified class map from individual masks.
    
    Args:
        masks: List of mask dictionaries with 'segmentation' and 'classification' keys
        image_shape: (height, width) of the output class map
        class_to_id: Optional mapping from class names to IDs
        
    Returns:
        2D class map where each pixel has its class ID
    """
    if class_to_id is None:
        class_to_id = {
            "water": 1,
            "vegetation": 2,
            "plantation": 3,
            "building": 4,
            "other": 0
        }
    
    h, w = image_shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    
    # Sort masks by confidence (lower first, so higher confidence overwrites)
    sorted_masks = sorted(
        masks, 
        key=lambda m: m.get("confidence", 0)
    )
    
    for mask_data in sorted_masks:
        mask = mask_data.get("segmentation")
        if mask is None:
            continue
            
        class_name = mask_data.get("classification", "other")
        class_id = class_to_id.get(class_name, 0)
        
        if class_id > 0:  # Don't overwrite with background
            class_map[mask] = class_id
    
    return class_map


def class_map_to_masks(
    class_map: np.ndarray,
    id_to_class: Optional[dict] = None
) -> list[dict]:
    """
    Convert a class map back to individual mask dictionaries.
    
    Args:
        class_map: 2D array with class IDs
        id_to_class: Optional mapping from IDs to class names
        
    Returns:
        List of mask dictionaries with 'segmentation' and 'classification' keys
    """
    if id_to_class is None:
        id_to_class = {
            1: "water",
            2: "vegetation",
            3: "plantation",
            4: "building"
        }
    
    masks = []
    
    for class_id, class_name in id_to_class.items():
        class_mask = class_map == class_id
        
        if class_mask.any():
            masks.append({
                "segmentation": class_mask,
                "classification": class_name,
                "area": int(class_mask.sum())
            })
    
    return masks
