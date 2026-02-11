"""
Utility functions for image processing and mask-to-polygon conversion.
"""

import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import json
from typing import Any


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes and convert to RGB numpy array.
    
    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    return np.array(image)


def resize_image_if_needed(
    image: np.ndarray, 
    max_size: int = 2048
) -> tuple[np.ndarray, float]:
    """
    Resize image if it exceeds the maximum dimension.
    
    Args:
        image: Input image array
        max_size: Maximum dimension (width or height)
        
    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_size:
        return image, 1.0
    
    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def mask_to_polygon(
    mask: np.ndarray, 
    simplify_tolerance: float = 2.0
) -> list[list[list[float]]]:
    """
    Convert a binary mask to polygon coordinates.
    
    Args:
        mask: Binary mask array (H, W)
        simplify_tolerance: Douglas-Peucker simplification tolerance in pixels
        
    Returns:
        List of polygon rings, each ring is a list of [x, y] coordinates
    """
    # Find contours in the mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return []
    
    polygons = []
    for contour in contours:
        # Flatten contour array and convert to list of [x, y]
        if len(contour) < 3:
            continue
            
        points = contour.reshape(-1, 2).tolist()
        
        # Create Shapely polygon for simplification
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygons
            
            # Simplify to reduce point count
            simplified = poly.simplify(simplify_tolerance, preserve_topology=True)
            
            if simplified.is_empty:
                continue
                
            if isinstance(simplified, MultiPolygon):
                for geom in simplified.geoms:
                    coords = list(geom.exterior.coords)
                    polygons.append(coords)
            else:
                coords = list(simplified.exterior.coords)
                polygons.append(coords)
                
        except Exception:
            # Fallback to raw contour points
            if len(points) >= 3:
                points.append(points[0])  # Close the ring
                polygons.append(points)
    
    return polygons


def pixel_coords_to_geo(
    pixel_coords: list[list[float]],
    bounds: dict[str, float],
    image_size: tuple[int, int]
) -> list[list[float]]:
    """
    Convert pixel coordinates to geographic coordinates.
    
    Args:
        pixel_coords: List of [x, y] pixel coordinates
        bounds: Dictionary with 'north', 'south', 'east', 'west' lat/lng bounds
        image_size: Tuple of (width, height) in pixels
        
    Returns:
        List of [longitude, latitude] coordinates
    """
    width, height = image_size
    west = bounds["west"]
    east = bounds["east"]
    north = bounds["north"]
    south = bounds["south"]
    
    geo_coords = []
    for x, y in pixel_coords:
        lng = west + (x / width) * (east - west)
        lat = north - (y / height) * (north - south)
        geo_coords.append([lng, lat])
    
    return geo_coords


def masks_to_geojson(
    masks: list[dict[str, Any]],
    bounds: dict[str, float],
    image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """
    Convert segmentation masks to GeoJSON polygons.
    
    Args:
        masks: List of mask dictionaries from segmentation
        bounds: Geographic bounds of the image
        image_size: Size of the image (width, height)
        
    Returns:
        List of GeoJSON Feature dictionaries
    """
    features = []
    
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        classification = mask_data.get("classification", "other")
        confidence = float(mask_data.get("stability_score", 0.0))
        
        # Convert mask to polygon(s)
        pixel_polygons = mask_to_polygon(mask)
        
        for polygon_coords in pixel_polygons:
            if len(polygon_coords) < 4:  # Need at least 3 points + closing
                continue
            
            # Convert to geographic coordinates
            geo_coords = pixel_coords_to_geo(polygon_coords, bounds, image_size)
            
            feature = {
                "type": "Feature",
                "properties": {
                    "classification": classification,
                    "confidence": round(confidence, 3),
                    "area_pixels": int(mask_data.get("area", 0))
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords]
                }
            }
            features.append(feature)
    
    return features


def merge_overlapping_masks(
    features: list[dict[str, Any]], 
    classification: str
) -> list[dict[str, Any]]:
    """
    Merge overlapping polygons of the same classification into a single feature.
    
    Args:
        features: List of GeoJSON features
        classification: Zone type to merge
        
    Returns:
        List of GeoJSON features with merged classification
    """
    from shapely.geometry import shape, mapping
    
    # Filter features by classification
    target_features = [f for f in features if f["properties"]["classification"] == classification]
    other_features = [f for f in features if f["properties"]["classification"] != classification]
    
    if len(target_features) == 0:
        return features
    
    if len(target_features) == 1:
        return features
    
    # Create Shapely polygons
    polygons = []
    confidences = []
    for f in target_features:
        try:
            poly = shape(f["geometry"])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
                confidences.append(f["properties"].get("confidence", 0))
        except Exception:
            continue
    
    if not polygons:
        return features
    
    # Merge overlapping polygons using unary_union
    merged = unary_union(polygons)
    
    # Create a single feature with the merged geometry
    if merged.is_empty:
        return other_features
    
    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Calculate total area from merged geometry
    total_area = sum(f["properties"].get("area_pixels", 0) for f in target_features)
    
    merged_feature = {
        "type": "Feature",
        "properties": {
            "classification": classification, 
            "confidence": round(avg_confidence, 3),
            "area_pixels": total_area
        },
        "geometry": mapping(merged)
    }
    
    return other_features + [merged_feature]


def resolve_class_overlaps(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Resolve overlaps between different classes using FIXED priority order.
    
    Priority order (water > building > plantation > vegetation):
    - Water ALWAYS wins - pools should never be classified as vegetation
    - This ensures consistent results regardless of confidence scores
    
    Args:
        features: List of GeoJSON features (should be one per class after merging)
        
    Returns:
        List of GeoJSON features with no inter-class overlaps
    """
    from shapely.geometry import shape, mapping
    from .config import CLASS_PRIORITY
    
    if len(features) <= 1:
        return features
    
    # Sort features by CLASS PRIORITY (not confidence) - higher priority wins
    sorted_features = sorted(
        features, 
        key=lambda f: CLASS_PRIORITY.get(f["properties"].get("classification", "other"), 0), 
        reverse=True
    )
    
    resolved_features = []
    accumulated_geometry = None  # Union of all higher-priority geometries
    
    for feature in sorted_features:
        try:
            classification = feature["properties"].get("classification", "other")
            feature_shape = shape(feature["geometry"])
            
            if not feature_shape.is_valid:
                feature_shape = feature_shape.buffer(0)
            
            if accumulated_geometry is not None:
                # Subtract all higher-priority geometries from this one
                feature_shape = feature_shape.difference(accumulated_geometry)
            
            if feature_shape.is_empty:
                continue
            
            # Update accumulated geometry for next iterations
            if accumulated_geometry is None:
                accumulated_geometry = feature_shape
            else:
                accumulated_geometry = accumulated_geometry.union(feature_shape)
            
            # Create the resolved feature
            resolved_feature = {
                "type": "Feature",
                "properties": feature["properties"].copy(),
                "geometry": mapping(feature_shape)
            }
            resolved_features.append(resolved_feature)
            
        except Exception as e:
            print(f"Warning: Failed to resolve overlap for {feature['properties'].get('classification')}: {e}")
            # Keep the original feature if resolution fails
            resolved_features.append(feature)
    
    return resolved_features


def clip_features_to_geometry(
    features: list[dict[str, Any]], 
    clip_geometry: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Clip segmentation features to a property boundary.
    
    Args:
        features: List of GeoJSON features to clip
        clip_geometry: GeoJSON geometry (Polygon or MultiPolygon) to clip to
        
    Returns:
        List of clipped GeoJSON features
    """
    from shapely.geometry import shape, mapping
    
    # Parse the clip geometry
    try:
        clip_shape = shape(clip_geometry)
        if not clip_shape.is_valid:
            clip_shape = clip_shape.buffer(0)
    except Exception as e:
        print(f"Warning: Invalid clip geometry: {e}")
        return features
    
    clipped_features = []
    
    for feature in features:
        try:
            feature_shape = shape(feature["geometry"])
            if not feature_shape.is_valid:
                feature_shape = feature_shape.buffer(0)
            
            # Intersect with clip geometry
            clipped = feature_shape.intersection(clip_shape)
            
            if clipped.is_empty:
                continue
            
            # Handle resulting geometry (may be MultiPolygon after clipping)
            if isinstance(clipped, (MultiPolygon, Polygon)):
                 if not clipped.is_empty and clipped.area > 0:
                     clipped_features.append({
                         "type": "Feature",
                         "properties": feature["properties"].copy(),
                         "geometry": mapping(clipped)
                     })
        except Exception as e:
            print(f"Warning: Failed to clip feature: {e}")
            continue
    
    return clipped_features
