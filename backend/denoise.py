"""OpenCV Non-Local Means denoising logic for Denoise Studio."""

import io
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from schemas import CropRegion

logger = logging.getLogger(__name__)

TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21
H_MIN = 1.0
H_MAX = 14.0
H_COLOR_MIN = 1.0
H_COLOR_MAX = 10.0


def _lerp(lo: float, hi: float, t: float) -> float:
    """Linear interpolate between lo and hi by factor t (0–1)."""
    return lo + (hi - lo) * max(0.0, min(1.0, t))


def _compute_nlm_params(luminance_strength: float, color_strength: float) -> Tuple[float, float]:
    """Map 0–1 slider values to NLM h and hColor parameters."""
    h = _lerp(H_MIN, H_MAX, luminance_strength)
    h_color = _lerp(H_COLOR_MIN, H_COLOR_MAX, color_strength)
    return h, h_color


def denoise_array(
    bgr: np.ndarray,
    luminance_strength: float,
    color_strength: float,
) -> np.ndarray:
    """Apply cv2.fastNlMeansDenoisingColored to a BGR uint8 array.

    Args:
        bgr: Input BGR uint8 image array.
        luminance_strength: Slider value 0.0–1.0 for luminance noise.
        color_strength: Slider value 0.0–1.0 for color noise.

    Returns:
        Denoised BGR uint8 array of the same shape.
    """
    h, h_color = _compute_nlm_params(luminance_strength, color_strength)
    logger.debug("NLM params: h=%.2f hColor=%.2f", h, h_color)

    result = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=h,
        hColor=h_color,
        templateWindowSize=TEMPLATE_WINDOW,
        searchWindowSize=SEARCH_WINDOW,
    )
    return result


def denoise_crops(
    bgr: np.ndarray,
    regions: List[CropRegion],
    luminance_strength: float,
    color_strength: float,
) -> List[np.ndarray]:
    """Denoise only the specified crop regions for preview mode.

    Args:
        bgr: Full-resolution BGR source image.
        regions: List of crop rectangles to process.
        luminance_strength: Slider value 0.0–1.0.
        color_strength: Slider value 0.0–1.0.

    Returns:
        List of denoised BGR crops, same length as regions.
    """
    results = []
    for region in regions:
        x, y, w, h = region.x, region.y, region.w, region.h
        img_h, img_w = bgr.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        crop = bgr[y:y + h, x:x + w].copy()
        denoised_crop = denoise_array(crop, luminance_strength, color_strength)
        results.append(denoised_crop)
    return results


def encode_image(
    bgr: np.ndarray,
    output_format: str,
    jpeg_quality: int = 95,
) -> Tuple[bytes, str]:
    """Encode a BGR array to bytes in the requested format.

    Args:
        bgr: BGR uint8 numpy array.
        output_format: 'jpeg' or 'tiff'.
        jpeg_quality: JPEG quality 1–100 (only used for JPEG output).

    Returns:
        Tuple of (raw bytes, media_type string).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    buf = io.BytesIO()
    if output_format == "tiff":
        pil_img.save(buf, format="TIFF", compression="tiff_lzw")
        media_type = "image/tiff"
    else:
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
        media_type = "image/jpeg"

    return buf.getvalue(), media_type


def encode_crop_jpeg(bgr_crop: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR crop to a base64 JPEG string for preview responses."""
    import base64
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
