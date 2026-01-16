"""
FastAPI application for ESVAL Segmentation Service.
Provides REST API endpoints for satellite image segmentation.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import json

from .config import get_device_info, MAX_IMAGE_SIZE
from .segmentation import get_segmentation_service, SegmentationService
from .utils import (
    load_image_from_bytes,
    resize_image_if_needed,
    masks_to_geojson,
    merge_overlapping_masks,
    clip_features_to_geometry
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the segmentation service on startup."""
    print("Loading MobileSAM model...")
    try:
        service = get_segmentation_service()
        print(f"Model ready on {service.device}")
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")
    yield


app = FastAPI(
    title="ESVAL Segmentation Service",
    description="MobileSAM-based image segmentation for satellite imagery",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    device: dict[str, Any]
    model_loaded: bool


class SegmentRequest(BaseModel):
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    bbox: list[float] | None = None


class BoundsInput(BaseModel):
    north: float
    south: float
    east: float
    west: float


class SegmentationResponse(BaseModel):
    features: list[dict[str, Any]]
    device_used: str
    processing_time_ms: float


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns service status and device capabilities.
    """
    try:
        service = get_segmentation_service()
        model_loaded = service.is_ready()
    except Exception:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        device=get_device_info(),
        model_loaded=model_loaded
    )


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    points: str | None = Form(None),
    point_labels: str | None = Form(None),
    bbox: str | None = Form(None),
    bounds: str | None = Form(None)
):
    """
    Segment image using point or bounding box prompts.
    
    Args:
        image: The satellite image to segment
        points: JSON array of [x, y] point coordinates
        point_labels: JSON array of labels (1=foreground, 0=background)
        bbox: JSON array [x1, y1, x2, y2] for bounding box prompt
        bounds: JSON object with geographic bounds {north, south, east, west}
    
    Returns:
        GeoJSON features with segmented polygons
    """
    import time
    start_time = time.time()
    
    try:
        service = get_segmentation_service()
        if not service.is_ready():
            raise HTTPException(503, "Model not loaded")
    except Exception as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")
    
    # Load and preprocess image
    image_bytes = await image.read()
    img_array = load_image_from_bytes(image_bytes)
    img_array, scale = resize_image_if_needed(img_array, MAX_IMAGE_SIZE)
    
    # Parse prompts
    parsed_points = json.loads(points) if points else None
    parsed_labels = json.loads(point_labels) if point_labels else None
    parsed_bbox = json.loads(bbox) if bbox else None
    parsed_bounds = json.loads(bounds) if bounds else None
    
    # Scale prompts if image was resized
    if scale != 1.0 and parsed_points:
        parsed_points = [[x * scale, y * scale] for x, y in parsed_points]
    if scale != 1.0 and parsed_bbox:
        parsed_bbox = [v * scale for v in parsed_bbox]
    
    # Run segmentation
    masks, scores = service.segment_with_prompts(
        img_array,
        points=parsed_points,
        point_labels=parsed_labels,
        box=parsed_bbox
    )
    
    # Convert to GeoJSON if bounds provided
    if parsed_bounds:
        h, w = img_array.shape[:2]
        # Take the best mask (highest score)
        best_idx = scores.argmax()
        mask_data = [{
            "segmentation": masks[best_idx],
            "stability_score": float(scores[best_idx]),
            "classification": "vegetation",  # Default, can be refined
            "area": masks[best_idx].sum()
        }]
        features = masks_to_geojson(mask_data, parsed_bounds, (w, h))
    else:
        # Return pixel coordinates
        features = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            features.append({
                "mask_index": i,
                "confidence": float(score),
                "area_pixels": int(mask.sum())
            })
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "features": features,
        "device_used": str(service.device),
        "processing_time_ms": round(processing_time, 2)
    }


@app.post("/auto-segment", response_model=SegmentationResponse)
async def auto_segment(
    image: UploadFile = File(...),
    bounds: str = Form(...),
    clip_geometry: str | None = Form(None)
):
    """
    Automatically segment all zones in the image.
    
    Args:
        image: The satellite image to segment
        bounds: JSON object with geographic bounds {north, south, east, west}
    
    Returns:
        GeoJSON features with classified zones (water, vegetation, plantation)
    """
    import time
    start_time = time.time()
    
    try:
        service = get_segmentation_service()
        if not service.is_ready():
            raise HTTPException(503, "Model not loaded")
    except Exception as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")
    
    # Load and preprocess image
    image_bytes = await image.read()
    img_array = load_image_from_bytes(image_bytes)
    original_h, original_w = img_array.shape[:2]
    img_array, scale = resize_image_if_needed(img_array, MAX_IMAGE_SIZE)
    
    # Parse bounds
    try:
        parsed_bounds = json.loads(bounds)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid bounds JSON")
    
    # Parse optional clip geometry
    parsed_clip_geometry = None
    if clip_geometry:
        try:
            parsed_clip_geometry = json.loads(clip_geometry)
        except json.JSONDecodeError:
            pass  # Ignore invalid clip geometry
    
    # Run auto-segmentation
    masks = service.auto_segment(img_array)
    
    # Convert to GeoJSON
    h, w = img_array.shape[:2]
    features = masks_to_geojson(masks, parsed_bounds, (w, h))
    
    # Merge overlapping masks of same type
    for zone_type in ["water", "vegetation", "plantation"]:
        features = merge_overlapping_masks(features, zone_type)
    
    # Resolve inter-class overlaps - higher confidence wins
    from app.utils import resolve_class_overlaps
    features = resolve_class_overlaps(features)
    
    # Clip features to property boundary if provided
    if parsed_clip_geometry:
        features = clip_features_to_geometry(features, parsed_clip_geometry)
    
    processing_time = (time.time() - start_time) * 1000
    
    return SegmentationResponse(
        features=features,
        device_used=str(service.device),
        processing_time_ms=round(processing_time, 2)
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "ESVAL Segmentation Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
