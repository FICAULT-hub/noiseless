"""Hybrid denoising pipeline: BM3D wavelet-domain + guided chroma filter + face adaptation.

Pipeline stages:
  0 – Preprocessing: decode → float32 RGB → LAB, separate L/A/B
  1 – Chroma denoising: BM3D on A/B + guided filter (edge-snapping from L)
  2 – Luminance denoising: 3-level wavelet decomposition, BM3D per subband
  3 – Detail recovery: re-add detail map where local variance > noise floor
  4 – Face-aware adaptation: lighter denoising + stronger detail in face regions
  5 – Perceptual sharpening: unsharp mask gated by local contrast
  6 – Recombine LAB → RGB, clip, return float32
"""

import io
import logging
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

# ── Optional heavy dependencies ──────────────────────────────────────────────
try:
    import bm3d as _bm3d
    _BM3D_AVAILABLE = True
except ImportError:
    _BM3D_AVAILABLE = False
    logger.warning("bm3d not available — Stage 2 will use NLM fallback")

try:
    import skimage.color as _skcolor
    import skimage.restoration as _skrest
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available — LAB conversion will use OpenCV")

try:
    import pywt as _pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False
    logger.warning("PyWavelets not available — Stage 2 will use NLM fallback")

try:
    import pillow_avif  # noqa: F401 — registers AVIF decoder
except ImportError:
    pass
try:
    import imagecodecs  # noqa: F401 — registers JXL decoder
except ImportError:
    pass


# ── Parameter mapping ────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate: a + (b-a)*t, with t clamped to [0, 1]."""
    return a + (b - a) * max(0.0, min(1.0, t))


def compute_params(luminance: float, color: float) -> Dict[str, object]:
    """Map slider values (0–1) to all internal algorithm parameters.

    Args:
        luminance: Luminance noise strength, 0.0 (minimal) to 1.0 (maximal).
        color: Color noise strength, 0.0 (minimal) to 1.0 (maximal).

    Returns:
        Dict keyed by parameter name; all values are floats or ints.
    """
    return {
        # Stage 1 — Chroma (BM3D on A/B + guided filter)
        "cbm3d_chroma_sigma":  _lerp(2.0, 25.0, color),
        "chroma_guided_radius": int(_lerp(4, 12, color)),
        # Inverse: higher color → lower eps → sharper guided-filter edges
        "chroma_guided_eps":   _lerp(0.01, 0.001, color),

        # Stage 2 — Luminance subbands
        "bm3d_LL_sigma":       _lerp(5.0, 30.0, luminance),
        "bm3d_LH_HL_sigma":    _lerp(2.0, 15.0, luminance),
        "nlm_HH_h":            _lerp(1.0, 6.0,  luminance),

        # Stage 3 — Detail recovery
        # Inverse: heavy denoising → less recovery to avoid re-introducing texture noise
        "detail_strength":         _lerp(0.8, 0.2, luminance),
        "noise_sigma_multiplier":  _lerp(1.0, 1.4, luminance),

        # Stage 4 — Face protection
        "face_strength_ratio": _lerp(0.5, 0.3, luminance),

        # Stage 5 — Perceptual sharpening
        # Inverse: heavy denoising → less USM to avoid halos on already-smoothed edges
        "usm_amount":    _lerp(0.25, 0.10, luminance),
        "usm_radius":    0.8,  # sub-pixel — never creates halos
        "usm_threshold": 3,    # skip flat areas to avoid amplifying residual noise
    }


# ── Image codec helpers ──────────────────────────────────────────────────────

def _decode_to_float_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to float32 RGB [0, 1].

    Uses Pillow (AVIF via pillow-avif-plugin, JXL via imagecodecs) and
    converts to RGB via .convert('RGB') to normalise any input mode.

    Args:
        image_bytes: Raw file bytes (JPEG, TIFF, AVIF, JXL, …).

    Returns:
        numpy array, shape (H, W, 3), dtype float32, range [0.0, 1.0].

    Raises:
        ValueError: If decoding fails.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8).astype(np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Image decode failed: {exc}") from exc


def bgr_uint8_to_float_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR uint8 array to float32 RGB [0, 1].

    Args:
        bgr: numpy array, shape (H, W, 3), dtype uint8, BGR channel order.

    Returns:
        numpy array, shape (H, W, 3), dtype float32, RGB channel order, [0, 1].
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
    return rgb.astype(np.float32) / 255.0        # (H, W, 3) float32


def encode(image_array: np.ndarray, output_format: str) -> bytes:
    """Encode a float32 [0, 1] RGB array to JPEG or TIFF bytes.

    Args:
        image_array: numpy array, shape (H, W, 3), float32, RGB, range [0, 1].
        output_format: 'jpeg' or 'tiff'.

    Returns:
        Encoded image bytes.
    """
    uint8 = (image_array * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8)
    buf = io.BytesIO()
    if output_format == "tiff":
        pil_img.save(buf, format="TIFF", compression="tiff_lzw")
    else:
        pil_img.save(buf, format="JPEG", quality=97, subsampling=0)
    return buf.getvalue()


# ── Wavelet / NLM helpers ────────────────────────────────────────────────────

def _bm3d_denoise(arr: np.ndarray, sigma_psd: float) -> np.ndarray:
    """Apply BM3D to a 2D float array with given sigma.

    Falls back to Gaussian smoothing if bm3d is unavailable.

    Args:
        arr: 2D float32 numpy array (any range).
        sigma_psd: Noise standard deviation in the same units as arr.

    Returns:
        Denoised array, same shape and dtype as arr.
    """
    if _BM3D_AVAILABLE:
        return _bm3d.bm3d(arr, sigma_psd=sigma_psd).astype(arr.dtype)
    # Gaussian fallback: radius proportional to sigma_psd scaled to [0,1]
    sigma_px = max(0.5, sigma_psd * 255.0 * 0.3)
    return scipy.ndimage.gaussian_filter(arr, sigma=sigma_px).astype(arr.dtype)


def _nlm_wavelet_subband(subband: np.ndarray, h: float) -> np.ndarray:
    """Apply NLM denoising to a wavelet detail subband.

    Detail subbands have small, possibly negative values — unsuitable for
    direct uint8 NLM. We rescale to [0, 255], denoise, then restore range.

    Args:
        subband: 2D float32 wavelet subband (may contain negative values).
        h: NLM filter strength parameter.

    Returns:
        Denoised subband, float32, same shape.
    """
    s_min = float(subband.min())
    s_max = float(subband.max())
    span  = s_max - s_min
    if span < 1e-10:
        return subband.copy()

    scaled   = ((subband - s_min) / span * 255.0).clip(0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(scaled, h=float(h))  # uint8 (H, W)
    return (denoised.astype(np.float32) / 255.0 * span + s_min)


# ── LAB conversion helpers ───────────────────────────────────────────────────

def _rgb_to_lab_norm(rgb: np.ndarray):
    """Convert float32 RGB [0,1] → LAB, normalised to [0,1] per channel.

    LAB natural ranges: L ∈ [0,100], A ∈ [-128,127], B ∈ [-128,127].
    Normalisation: L/100, (A+128)/255, (B+128)/255.

    Returns:
        Tuple (L_norm, A_norm, B_norm) each float32, shape (H, W), range [0,1].
    """
    if _SKIMAGE_AVAILABLE:
        lab = _skcolor.rgb2lab(rgb.astype(np.float64))  # (H,W,3) float64
    else:
        # OpenCV LAB: L∈[0,255], A∈[0,255], B∈[0,255] (shifted+scaled)
        bgr = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab_uint8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        # Undo OpenCV's encoding: L=L*100/255, A=A-128, B=B-128
        lab = np.zeros_like(lab_uint8)
        lab[:, :, 0] = lab_uint8[:, :, 0] * 100.0 / 255.0
        lab[:, :, 1] = lab_uint8[:, :, 1] - 128.0
        lab[:, :, 2] = lab_uint8[:, :, 2] - 128.0

    L_norm = (lab[:, :, 0] / 100.0).astype(np.float32)           # [0,1]
    A_norm = ((lab[:, :, 1] + 128.0) / 255.0).astype(np.float32) # [0,1]
    B_norm = ((lab[:, :, 2] + 128.0) / 255.0).astype(np.float32) # [0,1]
    return L_norm, A_norm, B_norm


def _lab_norm_to_rgb(L_norm, A_norm, B_norm) -> np.ndarray:
    """Denormalise [0,1] LAB channels and convert back to float32 RGB [0,1].

    Args:
        L_norm: float32 (H, W), range [0,1].
        A_norm: float32 (H, W), range [0,1].
        B_norm: float32 (H, W), range [0,1].

    Returns:
        float32 (H, W, 3) RGB, range [0,1].
    """
    L_out = (L_norm * 100.0).astype(np.float64)          # [0,100]
    A_out = (A_norm * 255.0 - 128.0).astype(np.float64)  # [-128,127]
    B_out = (B_norm * 255.0 - 128.0).astype(np.float64)  # [-128,127]
    lab = np.stack([L_out, A_out, B_out], axis=-1)        # (H,W,3) float64

    if _SKIMAGE_AVAILABLE:
        rgb = _skcolor.lab2rgb(lab)  # (H,W,3) float64 [0,1]
    else:
        # Re-encode into OpenCV LAB uint8
        lab_cv = np.zeros_like(lab, dtype=np.uint8)
        lab_cv[:, :, 0] = np.clip(lab[:, :, 0] * 255.0 / 100.0, 0, 255).astype(np.uint8)
        lab_cv[:, :, 1] = np.clip(lab[:, :, 1] + 128.0, 0, 255).astype(np.uint8)
        lab_cv[:, :, 2] = np.clip(lab[:, :, 2] + 128.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB).astype(np.float64) / 255.0

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ── Pipeline stages ──────────────────────────────────────────────────────────

def _stage1_chroma(L_norm, A_norm, B_norm, params):
    """BM3D on chroma A/B channels + guided filter edge-snapping from luminance.

    Returns:
        (clean_A, clean_B) float32 (H, W) arrays, range [0, 1].
        Falls back to original A/B unchanged on any error.
    """
    try:
        sigma = params["cbm3d_chroma_sigma"] / 255.0
        clean_A = _bm3d_denoise(A_norm, sigma)
        clean_B = _bm3d_denoise(B_norm, sigma)

        # Guided filter: use luminance as guide to snap chroma to luma edges
        clean_A = cv2.ximgproc.guidedFilter(
            guide=L_norm,
            src=clean_A,
            radius=params["chroma_guided_radius"],
            eps=params["chroma_guided_eps"],
            dDepth=-1,
        )
        clean_B = cv2.ximgproc.guidedFilter(
            guide=L_norm,
            src=clean_B,
            radius=params["chroma_guided_radius"],
            eps=params["chroma_guided_eps"],
            dDepth=-1,
        )
        return (
            np.clip(clean_A, 0.0, 1.0).astype(np.float32),
            np.clip(clean_B, 0.0, 1.0).astype(np.float32),
        )
    except Exception:
        logger.error("Stage 1 (chroma) failed — using original A/B: %s", traceback.format_exc())
        return A_norm.copy(), B_norm.copy()


def _stage2_luminance(L_norm, params) -> np.ndarray:
    """3-level wavelet decomposition with BM3D per subband + NLM on diagonal detail.

    Subband treatment (coarser → finer):
      LL3             : BM3D at full sigma (most visible noise here)
      LH3, HL3        : BM3D at full sigma
      HH3             : NLM (diagonal texture — BM3D over-smooths diagonals)
      LH2, HL2        : BM3D at 0.6× sigma
      HH2             : NLM at 0.6× h
      LH1, HL1        : BM3D at 0.3× sigma (fine detail — light touch)
      HH1             : untouched (finest diagonal, preserve entirely)

    Returns:
        Denoised L channel, float32, shape (H, W), range [0, 1].
        Falls back to NLM on full L channel on any error.
    """
    if not _PYWT_AVAILABLE:
        return _luminance_nlm_fallback(L_norm, h=8)

    try:
        # wavedec2 returns: [cA3, (cH3,cV3,cD3), (cH2,cV2,cD2), (cH1,cV1,cD1)]
        # Terminology: LL=cA, LH=cH (horizontal), HL=cV (vertical), HH=cD (diagonal)
        coeffs = _pywt.wavedec2(L_norm, wavelet='db4', level=3)
        LL3             = coeffs[0]           # float64 (H/8, W/8)
        LH3, HL3, HH3   = coeffs[1]           # float64 (H/8, W/8) each
        LH2, HL2, HH2   = coeffs[2]           # float64 (H/4, W/4) each
        LH1, HL1, HH1   = coeffs[3]           # float64 (H/2, W/2) each

        sigma_full = params["bm3d_LL_sigma"]    / 255.0
        sigma_mid  = params["bm3d_LH_HL_sigma"] / 255.0
        h_full     = params["nlm_HH_h"]

        # Level 3 (coarsest): full-strength BM3D + NLM on diagonal
        clean_LL3 = _bm3d_denoise(LL3.astype(np.float32), sigma_full).astype(np.float64)
        clean_LH3 = _bm3d_denoise(LH3.astype(np.float32), sigma_mid).astype(np.float64)
        clean_HL3 = _bm3d_denoise(HL3.astype(np.float32), sigma_mid).astype(np.float64)
        clean_HH3 = _nlm_wavelet_subband(HH3.astype(np.float32), h_full).astype(np.float64)

        # Level 2 (medium): 60% sigma
        sigma_mid2 = sigma_mid * 0.6
        h_mid2     = h_full   * 0.6
        clean_LH2 = _bm3d_denoise(LH2.astype(np.float32), sigma_mid2).astype(np.float64)
        clean_HL2 = _bm3d_denoise(HL2.astype(np.float32), sigma_mid2).astype(np.float64)
        clean_HH2 = _nlm_wavelet_subband(HH2.astype(np.float32), h_mid2).astype(np.float64)

        # Level 1 (finest): 30% sigma, leave HH1 completely untouched
        sigma_fine = sigma_mid * 0.3
        clean_LH1 = _bm3d_denoise(LH1.astype(np.float32), sigma_fine).astype(np.float64)
        clean_HL1 = _bm3d_denoise(HL1.astype(np.float32), sigma_fine).astype(np.float64)
        # HH1: finest diagonal detail — preserve entirely (no assignment)

        clean_coeffs = [
            clean_LL3,
            (clean_LH3, clean_HL3, clean_HH3),
            (clean_LH2, clean_HL2, clean_HH2),
            (clean_LH1, clean_HL1, HH1),   # HH1 unchanged
        ]
        reconstructed = _pywt.waverec2(clean_coeffs, wavelet='db4')

        # waverec2 may pad output by ≤1 pixel per level due to boundary handling
        img_h, img_w = L_norm.shape
        reconstructed = reconstructed[:img_h, :img_w]

        return np.clip(reconstructed, 0.0, 1.0).astype(np.float32)

    except Exception:
        logger.error("Stage 2 (wavelet) failed — NLM fallback: %s", traceback.format_exc())
        return _luminance_nlm_fallback(L_norm, h=8)


def _luminance_nlm_fallback(L_norm: np.ndarray, h: float) -> np.ndarray:
    """NLM on the full luminance channel as Stage 2 emergency fallback."""
    try:
        L_uint8   = (L_norm * 255.0).clip(0, 255).astype(np.uint8)
        denoised  = cv2.fastNlMeansDenoising(L_uint8, h=float(h))
        return denoised.astype(np.float32) / 255.0
    except Exception:
        logger.error("Stage 2 NLM fallback also failed")
        return L_norm.copy()


def _stage3_detail(L_norm, denoised_L, params) -> np.ndarray:
    """Re-add detail where local variance exceeds the noise floor.

    detail_map = original_L − Gaussian_blur(original_L)
    Only pixels where local_variance(detail_map) > noise_sigma² get detail back,
    preventing re-introduction of noise in flat areas.

    Returns:
        Recovered luminance, float32, shape (H, W), range [0, 1].
        Returns denoised_L unchanged on error.
    """
    try:
        blurred      = scipy.ndimage.gaussian_filter(L_norm.astype(np.float64), sigma=1.5)
        detail_map   = (L_norm.astype(np.float64) - blurred)  # float64 (H,W)

        local_var    = scipy.ndimage.uniform_filter(detail_map ** 2, size=5)  # float64

        if _SKIMAGE_AVAILABLE:
            noise_sigma = float(_skrest.estimate_sigma(L_norm))
        else:
            noise_sigma = float(np.std(L_norm)) * 0.1
        noise_sigma *= params["noise_sigma_multiplier"]

        # Mask: 1 where detail exceeds noise, 0 in flat/noisy areas
        detail_mask  = (local_var > noise_sigma ** 2).astype(np.float64)
        detail_mask  = scipy.ndimage.gaussian_filter(detail_mask, sigma=2.0)  # feather

        recovered = denoised_L + (
            detail_map * detail_mask * params["detail_strength"]
        ).astype(np.float32)
        return np.clip(recovered, 0.0, 1.0).astype(np.float32)

    except Exception:
        logger.error("Stage 3 (detail recovery) failed — skipping: %s", traceback.format_exc())
        return denoised_L.copy()


def _stage5_sharpen(adapted_L, params) -> np.ndarray:
    """Unsharp mask gated by local contrast.

    Only regions where |detail| > usm_threshold/255 are sharpened, leaving
    flat areas (which could contain noise) untouched.

    Returns:
        Sharpened luminance, float32, shape (H, W), range [0, 1].
        Returns adapted_L unchanged on error.
    """
    try:
        blurred       = scipy.ndimage.gaussian_filter(
            adapted_L.astype(np.float64), sigma=params["usm_radius"]
        )
        detail        = adapted_L.astype(np.float64) - blurred
        local_contrast = scipy.ndimage.uniform_filter(np.abs(detail), size=5)

        sharpen_mask  = (
            local_contrast > params["usm_threshold"] / 255.0
        ).astype(np.float64)

        sharpened = adapted_L + (
            detail * sharpen_mask * params["usm_amount"]
        ).astype(np.float32)
        return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

    except Exception:
        logger.error("Stage 5 (sharpening) failed — skipping: %s", traceback.format_exc())
        return adapted_L.copy()


# ── Public entry points ──────────────────────────────────────────────────────

def _run_stages(rgb_float: np.ndarray, params: dict) -> np.ndarray:
    """Execute all pipeline stages on a float32 RGB image.

    Args:
        rgb_float: Input, float32, shape (H, W, 3), range [0, 1], RGB.
        params: Parameter dict from compute_params().

    Returns:
        Denoised image, float32, shape (H, W, 3), range [0, 1], RGB.
    """
    img_h, img_w = rgb_float.shape[:2]
    mp = img_h * img_w / 1_000_000
    logger.info("Pipeline start: %dx%d (%.1fMP)", img_w, img_h, mp)

    # Stage 0 — Preprocessing: RGB → LAB, normalise each channel to [0,1]
    L_norm, A_norm, B_norm = _rgb_to_lab_norm(rgb_float)
    # L_norm: float32 (H,W) [0,1],  A/B_norm: float32 (H,W) [0,1]

    # Stage 1 — Chroma denoising
    clean_A, clean_B = _stage1_chroma(L_norm, A_norm, B_norm, params)

    # Stage 2 — Luminance wavelet denoising
    denoised_L = _stage2_luminance(L_norm, params)  # float32 (H,W) [0,1]

    # Stage 3 — Detail recovery
    recovered_L = _stage3_detail(L_norm, denoised_L, params)  # float32 (H,W) [0,1]

    # Stage 4 — Face-aware spatial adaptation
    try:
        if not _FACE_AWARE_AVAILABLE:
            raise ImportError("face_aware not loaded")
        adapted_L = _face_aware.apply(
            denoised_L=recovered_L,
            original_L=L_norm,
            params=params,
        )
    except Exception:
        logger.error("Stage 4 (face-aware) failed — skipping: %s", traceback.format_exc())
        adapted_L = recovered_L.copy()

    # Stage 5 — Perceptual sharpening
    sharpened_L = _stage5_sharpen(adapted_L, params)  # float32 (H,W) [0,1]

    # Stage 6 — Recombine LAB → RGB
    result = _lab_norm_to_rgb(sharpened_L, clean_A, clean_B)  # float32 (H,W,3) [0,1]
    logger.info("Pipeline complete")
    return result


def run(image_bytes: bytes, params: dict) -> np.ndarray:
    """Decode raw image bytes and run the full denoising pipeline.

    Args:
        image_bytes: Raw file bytes (JPEG, TIFF, AVIF, JXL).
        params: Parameter dict from compute_params().

    Returns:
        Denoised image, float32, shape (H, W, 3), range [0, 1], RGB.
    """
    rgb_float = _decode_to_float_rgb(image_bytes)
    return _run_stages(rgb_float, params)


def run_from_bgr(bgr: np.ndarray, params: dict) -> np.ndarray:
    """Run the pipeline on an OpenCV BGR uint8 array.

    Args:
        bgr: numpy array, shape (H, W, 3), dtype uint8, BGR channel order.
        params: Parameter dict from compute_params().

    Returns:
        Denoised image, float32, shape (H, W, 3), range [0, 1], RGB channel order.
    """
    rgb_float = bgr_uint8_to_float_rgb(bgr)
    return _run_stages(rgb_float, params)
