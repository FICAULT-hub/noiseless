"""Face-aware spatial adaptation for the denoising pipeline.

Detects face regions with Haar cascade and blends in a lighter,
detail-preserving denoising specifically tuned for skin and hair.
"""

import logging
import traceback

import cv2
import numpy as np
import scipy.ndimage

try:
    import bm3d as _bm3d_mod
    _BM3D_AVAILABLE = True
except ImportError:
    _BM3D_AVAILABLE = False

try:
    import skimage.restoration as _skrest
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


def apply(
    denoised_L: np.ndarray,
    original_L: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Detect face regions and blend spatially adaptive denoising.

    Face regions receive lighter denoising (controlled by face_strength_ratio)
    and stronger detail recovery (1.3× the background multiplier) to preserve
    skin texture, hair, and facial features.

    Args:
        denoised_L: Background-denoised luminance, float32, shape (H, W), range [0, 1].
        original_L: Original noisy luminance, float32, shape (H, W), range [0, 1].
        params: Pipeline parameter dict produced by pipeline.compute_params().

    Returns:
        Adapted luminance, float32, shape (H, W), range [0, 1].
        Returns denoised_L unchanged when no faces are detected or on any error.
    """
    try:
        img_h, img_w = denoised_L.shape

        # Haar cascade expects uint8 grayscale
        gray_uint8 = (denoised_L * 255.0).clip(0, 255).astype(np.uint8)  # (H, W) uint8

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray_uint8,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if not len(faces):
            return denoised_L

        output = denoised_L.copy()  # float32 (H, W) — accumulate per-face blends

        for fx, fy, fw, fh in faces:
            try:
                _apply_face_region(
                    output=output,
                    denoised_L=denoised_L,
                    original_L=original_L,
                    params=params,
                    fx=int(fx), fy=int(fy), fw=int(fw), fh=int(fh),
                    img_h=img_h, img_w=img_w,
                )
            except Exception:
                logger.warning(
                    "Face region (x=%d y=%d w=%d h=%d) skipped: %s",
                    fx, fy, fw, fh, traceback.format_exc(),
                )

        return output.astype(np.float32)

    except Exception:
        logger.error("face_aware.apply failed, returning denoised_L unchanged: %s",
                     traceback.format_exc())
        return denoised_L


def _apply_face_region(
    output: np.ndarray,
    denoised_L: np.ndarray,
    original_L: np.ndarray,
    params: dict,
    fx: int, fy: int, fw: int, fh: int,
    img_h: int, img_w: int,
) -> None:
    """Blend face-specific denoising into output for one detected face box.

    Modifies output in-place.

    Args:
        output: Accumulation array, float32, shape (H, W). Modified in-place.
        denoised_L: Background-denoised luminance, float32, shape (H, W).
        original_L: Original noisy luminance, float32, shape (H, W).
        params: Pipeline parameters.
        fx, fy, fw, fh: Raw face bounding box from detectMultiScale.
        img_h, img_w: Full image dimensions.
    """
    # Expand bounding box by 20% on each side, clamped to image bounds
    pad_x = int(fw * 0.20)
    pad_y = int(fh * 0.20)
    x  = max(0, fx - pad_x)
    y  = max(0, fy - pad_y)
    x2 = min(img_w, fx + fw + pad_x)
    y2 = min(img_h, fy + fh + pad_y)
    w, h = x2 - x, y2 - y

    # Feathered Gaussian mask: value=1 inside face box, tapering off at edges
    mask = np.zeros((img_h, img_w), dtype=np.float64)
    mask[y:y2, x:x2] = 1.0
    mask = scipy.ndimage.gaussian_filter(mask, sigma=min(h, w) * 0.15)
    m_max = mask.max()
    if m_max > 0:
        mask /= m_max  # normalize to [0, 1]

    face_patch    = denoised_L[y:y2, x:x2]  # float32 (h, w)
    orig_patch    = original_L[y:y2, x:x2]  # float32 (h, w)

    # Lighter BM3D pass for the face region
    face_sigma = params["bm3d_LL_sigma"] * params["face_strength_ratio"]
    if _BM3D_AVAILABLE:
        face_denoised = _bm3d_mod.bm3d(
            face_patch,
            sigma_psd=face_sigma / 255.0,
        ).astype(np.float32)  # (h, w) float32
    else:
        # Fallback: Gaussian smooth when BM3D unavailable
        face_denoised = scipy.ndimage.gaussian_filter(face_patch, sigma=1.0).astype(np.float32)

    # Recover more fine detail in the face region than background
    blurred_orig      = scipy.ndimage.gaussian_filter(orig_patch, sigma=1.5)
    face_detail_map   = orig_patch - blurred_orig  # float32 (h, w)
    face_local_var    = scipy.ndimage.uniform_filter(
        face_detail_map.astype(np.float64) ** 2, size=5
    )  # float64 (h, w)

    if _SKIMAGE_AVAILABLE:
        face_noise_sigma = float(_skrest.estimate_sigma(orig_patch))
    else:
        face_noise_sigma = float(np.std(orig_patch)) * 0.1

    face_detail_mask = (face_local_var > face_noise_sigma ** 2).astype(np.float64)
    face_detail_mask = scipy.ndimage.gaussian_filter(face_detail_mask, sigma=2.0)

    # 1.3× detail recovery multiplier vs background (spec value)
    face_recovered = face_denoised + (
        face_detail_map
        * face_detail_mask.astype(np.float32)
        * params["detail_strength"]
        * 1.3
    )
    face_recovered = np.clip(face_recovered, 0.0, 1.0).astype(np.float32)

    # Blend into output using the feathered mask
    mask_patch = mask[y:y2, x:x2].astype(np.float32)  # (h, w)
    output[y:y2, x:x2] = (
        face_recovered * mask_patch
        + output[y:y2, x:x2] * (1.0 - mask_patch)
    )
