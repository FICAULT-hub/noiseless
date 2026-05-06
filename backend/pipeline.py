"""Hybrid denoising pipeline — scalable to 100 MP+.

Algorithm overview (all stages are O(n) or O(n log n), no BM3D):
  0 – Preprocessing  : BGR/bytes → float32 RGB → LAB, split L/A/B
  1 – Chroma         : Gaussian pre-smooth + guided filter (edge-snapping from L)
  2 – Luminance      : skimage BayesShrink wavelet soft-thresholding
  3 – Detail recovery: re-add structure where local variance > noise floor
  4 – Face-aware     : lighter denoising + stronger detail for face regions
  5 – Sharpening     : unsharp mask gated by local contrast
  6 – Recombine      : LAB → RGB, clip, float32

Large images (> _MAX_DIRECT_MP) are split into _TILE_SIZE×_TILE_SIZE tiles
with _TILE_OVERLAP padding, so peak memory stays O(tile²) regardless of
input resolution.
"""

import io
import logging
import math
import traceback
from typing import Dict

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

try:
    import face_aware as _face_aware
    _FACE_AWARE_AVAILABLE = True
except Exception:
    _FACE_AWARE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Optional dependencies (all gracefully degrade) ───────────────────────────
try:
    import skimage.color as _skcolor
    import skimage.restoration as _skrest
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image unavailable — Stage 2 uses NLM fallback, LAB uses OpenCV")

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

# ── Tiling constants ─────────────────────────────────────────────────────────
# Below threshold → single pass.  Above → spatial tiles, O(tile²) memory.
_MAX_DIRECT_MP = 4        # MP; tiles always used above this
_TILE_SIZE     = 1024     # pixels per tile side (1 MP per tile)
# Overlap absorbs: wavelet boundary (db4 level-3 ≈ 24 px), guided-filter
# radius (12 px), Gaussian sigma-2 (6 px) → 48 px minimum; use 64 for margin.
_TILE_OVERLAP  = 64


# ── Parameter mapping ────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def compute_params(luminance: float, color: float) -> Dict[str, object]:
    """Map 0–1 slider values to internal algorithm parameters.

    Args:
        luminance: Luminance noise strength (0 = minimal, 1 = maximal).
        color:     Color noise strength    (0 = minimal, 1 = maximal).

    Returns:
        Dict of named parameters consumed by each pipeline stage.
    """
    return {
        # Stage 1 — Chroma
        # chroma_gauss_sigma: Gaussian pre-smooth pixel sigma (0.25 → 3.1 px)
        "chroma_gauss_sigma":   _lerp(2.0, 25.0, color) / 8.0,
        "chroma_guided_radius": int(_lerp(4, 12, color)),
        # Inverse: higher color → lower eps → sharper edge-snapping
        "chroma_guided_eps":    _lerp(0.01, 0.001, color),

        # Stage 2 — Luminance wavelet (BayesShrink sigma in [0,1] image units)
        "wavelet_sigma":        _lerp(5.0, 30.0, luminance) / 255.0,

        # Stage 3 — Detail recovery
        "detail_strength":         _lerp(0.8, 0.2, luminance),   # inverse
        "noise_sigma_multiplier":  _lerp(1.0, 1.4, luminance),

        # Stage 4 — Face protection
        "face_strength_ratio":  _lerp(0.5, 0.3, luminance),

        # Stage 5 — Perceptual sharpening
        "usm_amount":    _lerp(0.25, 0.10, luminance),   # inverse
        "usm_radius":    0.8,   # sub-pixel, no halos
        "usm_threshold": 3,     # skip flat areas

        # Legacy keys kept so face_aware.py still reads them without change
        "bm3d_LL_sigma":       _lerp(5.0, 30.0, luminance),
        "detail_strength":     _lerp(0.8, 0.2, luminance),
    }


# ── Codec helpers ─────────────────────────────────────────────────────────────

def _decode_to_float_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes (JPEG/TIFF/AVIF/…) to float32 RGB [0, 1]."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8).astype(np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Image decode failed: {exc}") from exc


def bgr_uint8_to_float_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR uint8 → float32 RGB [0, 1]."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def encode(image_array: np.ndarray, output_format: str) -> bytes:
    """Encode float32 RGB [0, 1] → JPEG or TIFF bytes."""
    uint8   = (image_array * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8)
    buf     = io.BytesIO()
    if output_format == "tiff":
        pil_img.save(buf, format="TIFF", compression="tiff_lzw")
    else:
        pil_img.save(buf, format="JPEG", quality=97, subsampling=0)
    return buf.getvalue()


# ── LAB helpers ───────────────────────────────────────────────────────────────

def _rgb_to_lab_norm(rgb: np.ndarray):
    """float32 RGB [0,1] → (L_norm, A_norm, B_norm) each float32 [0,1].

    LAB natural ranges: L∈[0,100], A/B∈[-128,127].
    Normalisation: L/100, (A+128)/255, (B+128)/255.
    """
    if _SKIMAGE_AVAILABLE:
        lab = _skcolor.rgb2lab(rgb.astype(np.float64))
    else:
        bgr      = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab_u8   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        lab      = np.empty_like(lab_u8)
        lab[:, :, 0] = lab_u8[:, :, 0] * 100.0 / 255.0
        lab[:, :, 1] = lab_u8[:, :, 1] - 128.0
        lab[:, :, 2] = lab_u8[:, :, 2] - 128.0

    L = (lab[:, :, 0] / 100.0).astype(np.float32)
    A = ((lab[:, :, 1] + 128.0) / 255.0).astype(np.float32)
    B = ((lab[:, :, 2] + 128.0) / 255.0).astype(np.float32)
    return L, A, B


def _lab_norm_to_rgb(L_norm, A_norm, B_norm) -> np.ndarray:
    """(L_norm, A_norm, B_norm) float32 [0,1] → float32 RGB [0,1]."""
    L_out = (L_norm * 100.0).astype(np.float64)
    A_out = (A_norm * 255.0 - 128.0).astype(np.float64)
    B_out = (B_norm * 255.0 - 128.0).astype(np.float64)
    lab   = np.stack([L_out, A_out, B_out], axis=-1)

    if _SKIMAGE_AVAILABLE:
        rgb = _skcolor.lab2rgb(lab)
    else:
        lab_cv        = np.zeros_like(lab, dtype=np.uint8)
        lab_cv[:,:,0] = np.clip(lab[:,:,0] * 255.0 / 100.0, 0, 255).astype(np.uint8)
        lab_cv[:,:,1] = np.clip(lab[:,:,1] + 128.0,         0, 255).astype(np.uint8)
        lab_cv[:,:,2] = np.clip(lab[:,:,2] + 128.0,         0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB).astype(np.float64) / 255.0

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _stage1_chroma(L_norm, A_norm, B_norm, params):
    """Chroma denoising: Gaussian pre-smooth + guided filter.

    Gaussian removes high-frequency color noise; guided filter re-introduces
    luminance-edge structure so colour doesn't bleed across edges.

    Returns (clean_A, clean_B) float32 [0,1], falls back to original on error.
    """
    try:
        s = float(params["chroma_gauss_sigma"])
        clean_A = scipy.ndimage.gaussian_filter(A_norm.astype(np.float64), sigma=s).astype(np.float32)
        clean_B = scipy.ndimage.gaussian_filter(B_norm.astype(np.float64), sigma=s).astype(np.float32)

        r   = int(params["chroma_guided_radius"])
        eps = float(params["chroma_guided_eps"])
        clean_A = cv2.ximgproc.guidedFilter(guide=L_norm, src=clean_A, radius=r, eps=eps, dDepth=-1)
        clean_B = cv2.ximgproc.guidedFilter(guide=L_norm, src=clean_B, radius=r, eps=eps, dDepth=-1)

        return (np.clip(clean_A, 0.0, 1.0).astype(np.float32),
                np.clip(clean_B, 0.0, 1.0).astype(np.float32))
    except Exception:
        logger.error("Stage 1 (chroma) failed — original A/B: %s", traceback.format_exc())
        return A_norm.copy(), B_norm.copy()


def _stage2_luminance(L_norm, params) -> np.ndarray:
    """Luminance denoising via BayesShrink wavelet soft-thresholding.

    skimage.restoration.denoise_wavelet decomposes L into db4 subbands,
    computes per-subband optimal thresholds (BayesShrink minimises MSE),
    applies soft thresholding, and reconstructs.  Memory is O(image),
    speed is O(n log n) — scales cleanly to 100 MP tiles.

    Falls back to NLM on the full tile when scikit-image is unavailable.
    """
    try:
        if not _SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image unavailable")

        sigma = float(params["wavelet_sigma"])   # [0.02, 0.12] for slider 0→1
        denoised = _skrest.denoise_wavelet(
            L_norm,
            sigma=sigma,
            wavelet="db4",
            mode="soft",
            wavelet_levels=3,
            method="BayesShrink",
            rescale_sigma=True,
            channel_axis=None,
        )
        return np.clip(denoised, 0.0, 1.0).astype(np.float32)

    except Exception:
        logger.error("Stage 2 (wavelet) failed — NLM fallback: %s", traceback.format_exc())
        return _luminance_nlm_fallback(L_norm, h=float(params.get("wavelet_sigma", 0.04)) * 255 * 0.4)


def _luminance_nlm_fallback(L_norm: np.ndarray, h: float) -> np.ndarray:
    """NLM on the luminance tile — emergency fallback when skimage is missing."""
    try:
        uint8    = (L_norm * 255.0).clip(0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(uint8, h=max(1.0, float(h)))
        return denoised.astype(np.float32) / 255.0
    except Exception:
        logger.error("Stage 2 NLM fallback failed — returning L unchanged")
        return L_norm.copy()


def _stage3_detail(L_norm, denoised_L, params) -> np.ndarray:
    """Re-add detail where local variance exceeds the noise floor.

    Computes detail_map = L_norm − Gaussian(L_norm, σ=1.5).
    Pixels where variance(detail_map) > estimated_noise² are real structure;
    those get params["detail_strength"] of their detail re-added.
    A feathered mask avoids hard boundaries.
    """
    try:
        blurred    = scipy.ndimage.gaussian_filter(L_norm.astype(np.float64), sigma=1.5)
        detail_map = L_norm.astype(np.float64) - blurred
        local_var  = scipy.ndimage.uniform_filter(detail_map ** 2, size=5)

        if _SKIMAGE_AVAILABLE:
            noise_sigma = float(_skrest.estimate_sigma(L_norm))
        else:
            noise_sigma = float(np.std(L_norm)) * 0.1
        noise_sigma *= float(params["noise_sigma_multiplier"])

        mask = (local_var > noise_sigma ** 2).astype(np.float64)
        mask = scipy.ndimage.gaussian_filter(mask, sigma=2.0)

        recovered = denoised_L + (detail_map * mask * params["detail_strength"]).astype(np.float32)
        return np.clip(recovered, 0.0, 1.0).astype(np.float32)
    except Exception:
        logger.error("Stage 3 (detail) failed — skipping: %s", traceback.format_exc())
        return denoised_L.copy()


def _stage5_sharpen(adapted_L, params) -> np.ndarray:
    """Unsharp mask gated by local contrast.

    Skips flat areas (noise floor or sky) where USM would amplify residual noise.
    """
    try:
        blurred  = scipy.ndimage.gaussian_filter(adapted_L.astype(np.float64),
                                                  sigma=float(params["usm_radius"]))
        detail   = adapted_L.astype(np.float64) - blurred
        lc       = scipy.ndimage.uniform_filter(np.abs(detail), size=5)
        mask     = (lc > params["usm_threshold"] / 255.0).astype(np.float64)
        sharpened = adapted_L + (detail * mask * params["usm_amount"]).astype(np.float32)
        return np.clip(sharpened, 0.0, 1.0).astype(np.float32)
    except Exception:
        logger.error("Stage 5 (sharpen) failed — skipping: %s", traceback.format_exc())
        return adapted_L.copy()


# ── Core dispatcher ───────────────────────────────────────────────────────────

def _run_stages_direct(rgb_float: np.ndarray, params: dict) -> np.ndarray:
    """Run all six stages on one tile (or the whole image if small enough).

    Args:
        rgb_float: float32 (H, W, 3) RGB [0, 1].
        params:    dict from compute_params().

    Returns:
        float32 (H, W, 3) RGB [0, 1].
    """
    img_h, img_w = rgb_float.shape[:2]
    logger.info("Pipeline tile: %dx%d (%.1fMP)", img_w, img_h, img_h * img_w / 1e6)

    # Stage 0 — LAB decomposition
    L_norm, A_norm, B_norm = _rgb_to_lab_norm(rgb_float)

    # Stage 1 — Chroma
    clean_A, clean_B = _stage1_chroma(L_norm, A_norm, B_norm, params)

    # Stage 2 — Luminance wavelet
    denoised_L = _stage2_luminance(L_norm, params)

    # Stage 3 — Detail recovery
    recovered_L = _stage3_detail(L_norm, denoised_L, params)

    # Stage 4 — Face-aware adaptation
    adapted_L = recovered_L
    try:
        if not _FACE_AWARE_AVAILABLE:
            raise ImportError("face_aware not loaded")
        adapted_L = _face_aware.apply(denoised_L=recovered_L, original_L=L_norm, params=params)
    except Exception:
        logger.error("Stage 4 (face-aware) failed — skipping: %s", traceback.format_exc())

    # Stage 5 — Sharpening
    sharpened_L = _stage5_sharpen(adapted_L, params)

    # Stage 6 — Recombine
    return _lab_norm_to_rgb(sharpened_L, clean_A, clean_B)


def _run_stages_tiled(rgb_float: np.ndarray, params: dict) -> np.ndarray:
    """Process large image in _TILE_SIZE×_TILE_SIZE tiles with _TILE_OVERLAP padding.

    Peak memory is O(_TILE_SIZE²) regardless of input resolution.
    """
    img_h, img_w = rgb_float.shape[:2]
    result = np.empty_like(rgb_float)
    total  = math.ceil(img_h / _TILE_SIZE) * math.ceil(img_w / _TILE_SIZE)
    idx    = 0

    y = 0
    while y < img_h:
        y_end = min(y + _TILE_SIZE, img_h)
        x = 0
        while x < img_w:
            x_end  = min(x + _TILE_SIZE, img_w)
            idx   += 1
            logger.info("Pipeline tile %d/%d", idx, total)

            py0 = max(0, y - _TILE_OVERLAP);  py1 = min(img_h, y_end + _TILE_OVERLAP)
            px0 = max(0, x - _TILE_OVERLAP);  px1 = min(img_w, x_end + _TILE_OVERLAP)

            tile    = rgb_float[py0:py1, px0:px1]
            out     = _run_stages_direct(tile, params)

            iy0 = y - py0;  ix0 = x - px0
            result[y:y_end, x:x_end] = out[iy0:iy0 + (y_end - y), ix0:ix0 + (x_end - x)]
            x = x_end
        y = y_end

    return result


def _run_stages(rgb_float: np.ndarray, params: dict) -> np.ndarray:
    """Entry point: dispatch to tiled or direct based on image size."""
    img_h, img_w = rgb_float.shape[:2]
    mp = img_h * img_w / 1_000_000
    logger.info("Pipeline start: %dx%d (%.1fMP)", img_w, img_h, mp)
    if img_h * img_w <= _MAX_DIRECT_MP * 1_000_000:
        return _run_stages_direct(rgb_float, params)
    return _run_stages_tiled(rgb_float, params)


# ── Public API ────────────────────────────────────────────────────────────────

def run(image_bytes: bytes, params: dict) -> np.ndarray:
    """Decode image bytes and run the full pipeline.

    Args:
        image_bytes: Raw file bytes (JPEG, TIFF, AVIF, …).
        params: dict from compute_params().

    Returns:
        float32 (H, W, 3) RGB [0, 1].
    """
    return _run_stages(_decode_to_float_rgb(image_bytes), params)


def run_from_bgr(bgr: np.ndarray, params: dict) -> np.ndarray:
    """Run the pipeline on an OpenCV BGR uint8 array.

    Args:
        bgr: uint8 (H, W, 3) BGR.
        params: dict from compute_params().

    Returns:
        float32 (H, W, 3) RGB [0, 1].
    """
    return _run_stages(bgr_uint8_to_float_rgb(bgr), params)
