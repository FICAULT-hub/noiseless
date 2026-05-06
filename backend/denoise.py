"""Thin wrapper: delegates to pipeline.py while preserving the interface used by main.py.

Public API (unchanged from original):
    denoise_array(bgr, luminance_strength, color_strength) -> np.ndarray (BGR uint8)
    denoise_crops(bgr, regions, luminance_strength, color_strength) -> List[np.ndarray]
    encode_image(bgr, output_format, jpeg_quality) -> Tuple[bytes, str]
    encode_crop_jpeg(bgr_crop, quality) -> str
"""

import base64
import io
import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import pipeline
from schemas import CropRegion

logger = logging.getLogger(__name__)


def denoise_array(
    bgr: np.ndarray,
    luminance_strength: float,
    color_strength: float,
) -> np.ndarray:
    """Denoise a full image using the hybrid pipeline.

    Args:
        bgr: Input image, uint8, shape (H, W, 3), BGR channel order.
        luminance_strength: Slider value 0.0–1.0 for luminance noise.
        color_strength: Slider value 0.0–1.0 for color noise.

    Returns:
        Denoised image, uint8, shape (H, W, 3), BGR channel order.
    """
    params      = pipeline.compute_params(luminance_strength, color_strength)
    rgb_float   = pipeline.run_from_bgr(bgr, params)   # float32 (H,W,3) [0,1] RGB

    rgb_uint8   = (rgb_float * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)   # uint8 (H,W,3) BGR


def denoise_crops(
    bgr: np.ndarray,
    regions: List[CropRegion],
    luminance_strength: float,
    color_strength: float,
) -> List[np.ndarray]:
    """Denoise specific crop regions for preview mode.

    Computes pipeline parameters once, then runs the pipeline independently
    on each crop for efficiency.

    Args:
        bgr: Full-resolution source image, uint8, shape (H, W, 3), BGR.
        regions: List of crop rectangles to process.
        luminance_strength: Slider value 0.0–1.0.
        color_strength: Slider value 0.0–1.0.

    Returns:
        List of denoised BGR uint8 crops, same length as regions.
    """
    params  = pipeline.compute_params(luminance_strength, color_strength)
    img_h, img_w = bgr.shape[:2]
    results = []

    for region in regions:
        x = max(0, min(region.x, img_w - 1))
        y = max(0, min(region.y, img_h - 1))
        w = min(region.w, img_w - x)
        h = min(region.h, img_h - y)

        crop        = bgr[y:y + h, x:x + w].copy()          # uint8 BGR
        rgb_float   = pipeline.run_from_bgr(crop, params)    # float32 RGB
        rgb_uint8   = (rgb_float * 255.0).clip(0, 255).astype(np.uint8)
        results.append(cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

    return results


def encode_image(
    bgr: np.ndarray,
    output_format: str,
    jpeg_quality: int = 95,
) -> Tuple[bytes, str]:
    """Encode a BGR uint8 array to JPEG or TIFF bytes.

    Args:
        bgr: uint8 numpy array, shape (H, W, 3), BGR channel order.
        output_format: 'jpeg' or 'tiff'.
        jpeg_quality: JPEG quality 1–100 (ignored for TIFF).

    Returns:
        Tuple of (raw bytes, media_type string).
    """
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf     = io.BytesIO()

    if output_format == "tiff":
        pil_img.save(buf, format="TIFF", compression="tiff_lzw")
        return buf.getvalue(), "image/tiff"

    pil_img.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
    return buf.getvalue(), "image/jpeg"


def encode_crop_jpeg(bgr_crop: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR crop to a base64 JPEG string for preview responses.

    Args:
        bgr_crop: uint8 numpy array, shape (H, W, 3), BGR channel order.
        quality: JPEG quality 1–100.

    Returns:
        Base64-encoded JPEG string.
    """
    rgb     = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf     = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
