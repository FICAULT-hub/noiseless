"""OpenCV Non-Local Means denoising logic for Denoise Studio."""

import io
import logging
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from schemas import CropRegion

logger = logging.getLogger(__name__)

TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21
# Tiles keep peak memory O(tile²) regardless of image size.
# Overlap = SEARCH_WINDOW so every border pixel has full NLM context.
TILE_SIZE = 1024
TILE_OVERLAP = SEARCH_WINDOW
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


def _nlm(bgr: np.ndarray, h_param: float, h_color: float) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=h_param,
        hColor=h_color,
        templateWindowSize=TEMPLATE_WINDOW,
        searchWindowSize=SEARCH_WINDOW,
    )


def _denoise_tiled(bgr: np.ndarray, h_param: float, h_color: float) -> np.ndarray:
    """Tile-based NLM: process TILE_SIZE×TILE_SIZE chunks with TILE_OVERLAP padding
    so border pixels have full search context. Peak memory is O(tile²), not O(image²)."""
    img_h, img_w = bgr.shape[:2]
    result = np.empty_like(bgr)
    total = math.ceil(img_h / TILE_SIZE) * math.ceil(img_w / TILE_SIZE)
    idx = 0

    y = 0
    while y < img_h:
        y_end = min(y + TILE_SIZE, img_h)
        x = 0
        while x < img_w:
            x_end = min(x + TILE_SIZE, img_w)
            idx += 1
            logger.info("NLM tile %d/%d (%dx%d)", idx, total, x_end - x, y_end - y)

            pad_x0 = max(0, x - TILE_OVERLAP)
            pad_y0 = max(0, y - TILE_OVERLAP)
            pad_x1 = min(img_w, x_end + TILE_OVERLAP)
            pad_y1 = min(img_h, y_end + TILE_OVERLAP)

            denoised_padded = _nlm(bgr[pad_y0:pad_y1, pad_x0:pad_x1], h_param, h_color)

            inner_y0 = y - pad_y0
            inner_x0 = x - pad_x0
            result[y:y_end, x:x_end] = denoised_padded[
                inner_y0 : inner_y0 + (y_end - y),
                inner_x0 : inner_x0 + (x_end - x),
            ]
            x = x_end
        y = y_end

    return result


def denoise_array(
    bgr: np.ndarray,
    luminance_strength: float,
    color_strength: float,
) -> np.ndarray:
    h_param, h_color = _compute_nlm_params(luminance_strength, color_strength)
    img_h, img_w = bgr.shape[:2]
    mp = (img_h * img_w) / 1_000_000
    logger.info("NLM h=%.2f hColor=%.2f image=%dx%d (%.1fMP)", h_param, h_color, img_w, img_h, mp)

    if img_h <= TILE_SIZE and img_w <= TILE_SIZE:
        return _nlm(bgr, h_param, h_color)

    return _denoise_tiled(bgr, h_param, h_color)


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
