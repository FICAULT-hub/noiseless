"""Smart crop zone detection for Denoise Studio.

Detects three zones in an image:
  1. Highlights — brightest recoverable region
  2. Shadows / Blacks — darkest region with detail
  3. Faces / People — detected via Haar cascade, falls back to center crop
"""

import base64
import io
import logging
import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from schemas import AnalyzedZone

logger = logging.getLogger(__name__)

CROP_SIZE = 600
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
GRID_ROWS = 4
GRID_COLS = 4


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 array to LAB colorspace."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def _clamp_region(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Clamp a crop region to image bounds."""
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h


def _crop_to_base64(bgr: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Crop a BGR image and return a base64-encoded JPEG string."""
    crop = bgr[y:y + h, x:x + w]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _find_highlights_zone(lab: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Locate the brightest recoverable grid cell (L between 200–254 in uint8 LAB).

    Avoids completely blown-out regions (L == 255 in all pixels).
    Falls back to the globally brightest cell if no perfect match found.
    """
    l_channel = lab[:, :, 0].astype(np.float32)
    cell_h = img_h // GRID_ROWS
    cell_w = img_w // GRID_COLS

    best_score = -1.0
    best_cell = (0, 0)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cy = row * cell_h
            cx = col * cell_w
            cell = l_channel[cy:cy + cell_h, cx:cx + cell_w]
            mean_l = float(np.mean(cell))
            blown = float(np.mean(cell >= 254))
            if blown < 0.8:
                score = mean_l * (1.0 - blown)
                if score > best_score:
                    best_score = score
                    best_cell = (cy, cx)

    cy, cx = best_cell
    cx_center = cx + cell_w // 2
    cy_center = cy + cell_h // 2
    half = CROP_SIZE // 2
    x = max(0, cx_center - half)
    y = max(0, cy_center - half)
    return _clamp_region(x, y, CROP_SIZE, CROP_SIZE, img_w, img_h)


def _find_shadows_zone(lab: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Locate the darkest grid cell with recoverable shadow detail (L between 5–40 in 0–255 uint8).

    Falls back to the darkest cell overall if no cell meets the range.
    """
    l_channel = lab[:, :, 0].astype(np.float32)
    # LAB L channel in OpenCV uint8 is scaled 0–255 (corresponds to 0–100 in standard LAB)
    # 5–40 in standard LAB → 12.75–102 in uint8 (approx 13–102)
    SHADOW_LOW = 13.0
    SHADOW_HIGH = 102.0

    cell_h = img_h // GRID_ROWS
    cell_w = img_w // GRID_COLS

    best_score = float("inf")
    fallback_score = float("inf")
    best_cell = (0, 0)
    fallback_cell = (0, 0)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cy = row * cell_h
            cx = col * cell_w
            cell = l_channel[cy:cy + cell_h, cx:cx + cell_w]
            mean_l = float(np.mean(cell))
            if SHADOW_LOW <= mean_l <= SHADOW_HIGH:
                if mean_l < best_score:
                    best_score = mean_l
                    best_cell = (cy, cx)
            if mean_l < fallback_score:
                fallback_score = mean_l
                fallback_cell = (cy, cx)

    chosen = best_cell if best_score < float("inf") else fallback_cell
    cy, cx = chosen
    cx_center = cx + cell_w // 2
    cy_center = cy + cell_h // 2
    half = CROP_SIZE // 2
    x = max(0, cx_center - half)
    y = max(0, cy_center - half)
    return _clamp_region(x, y, CROP_SIZE, CROP_SIZE, img_w, img_h)


def _find_face_zone(bgr: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int, str]:
    """Detect a face using Haar cascade.

    Returns (x, y, w, h, label) where label is 'faces' or 'center'.
    Falls back to a center crop if no face is detected.
    """
    if not os.path.exists(CASCADE_PATH):
        logger.warning("Haar cascade not found at %s — using center crop", CASCADE_PATH)
        return _center_crop(img_w, img_h) + ("center",)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Scale down detection for speed on large images
    scale = min(1.0, 1200.0 / max(img_w, img_h))
    detect_w = int(img_w * scale)
    detect_h = int(img_h * scale)
    gray_small = cv2.resize(gray, (detect_w, detect_h)) if scale < 1.0 else gray

    faces = cascade.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return _center_crop(img_w, img_h) + ("center",)

    # Pick the largest detected face
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    if scale < 1.0:
        fx = int(fx / scale)
        fy = int(fy / scale)
        fw = int(fw / scale)
        fh = int(fh / scale)

    cx = fx + fw // 2
    cy = fy + fh // 2
    half = CROP_SIZE // 2
    x = max(0, cx - half)
    y = max(0, cy - half)
    x, y, w, h = _clamp_region(x, y, CROP_SIZE, CROP_SIZE, img_w, img_h)
    return x, y, w, h, "faces"


def _center_crop(img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Return a centered CROP_SIZE crop."""
    half = CROP_SIZE // 2
    cx = img_w // 2
    cy = img_h // 2
    x = max(0, cx - half)
    y = max(0, cy - half)
    w = min(CROP_SIZE, img_w - x)
    h = min(CROP_SIZE, img_h - y)
    return x, y, w, h


def detect_zones(bgr: np.ndarray) -> List[AnalyzedZone]:
    """Detect highlights, shadows, and face/center zones in a BGR image.

    Args:
        bgr: OpenCV BGR uint8 numpy array.

    Returns:
        List of three AnalyzedZone objects.
    """
    img_h, img_w = bgr.shape[:2]
    lab = _bgr_to_lab(bgr)

    hx, hy, hw, hh = _find_highlights_zone(lab, img_w, img_h)
    sx, sy, sw, sh = _find_shadows_zone(lab, img_w, img_h)
    fx, fy, fw, fh, face_label = _find_face_zone(bgr, img_w, img_h)

    zones = [
        AnalyzedZone(
            x=hx, y=hy, w=hw, h=hh,
            label="highlights",
            thumbnail_base64=_crop_to_base64(bgr, hx, hy, hw, hh),
        ),
        AnalyzedZone(
            x=sx, y=sy, w=sw, h=sh,
            label="shadows",
            thumbnail_base64=_crop_to_base64(bgr, sx, sy, sw, sh),
        ),
        AnalyzedZone(
            x=fx, y=fy, w=fw, h=fh,
            label=face_label,
            thumbnail_base64=_crop_to_base64(bgr, fx, fy, fw, fh),
        ),
    ]
    return zones
