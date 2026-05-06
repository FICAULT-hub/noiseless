"""Face-aware spatial adaptation for the denoising pipeline.

Detects face regions with Haar cascade and blends in lighter denoising
(wavelet thresholding at reduced sigma) with stronger detail recovery to
preserve skin texture, hair, and facial structure.
"""

import logging
import traceback

import cv2
import numpy as np
import scipy.ndimage

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
    """Detect faces and blend spatially-adaptive denoising.

    Face regions receive lighter denoising (face_strength_ratio × wavelet_sigma)
    and 1.3× stronger detail recovery vs background to preserve fine texture.

    Args:
        denoised_L: Background-denoised luminance, float32 (H, W), [0, 1].
        original_L: Original noisy luminance,      float32 (H, W), [0, 1].
        params:     Pipeline dict from pipeline.compute_params().

    Returns:
        Adapted luminance, float32 (H, W), [0, 1].
        Returns denoised_L unchanged when no faces detected or on any error.
    """
    try:
        img_h, img_w = denoised_L.shape
        gray = (denoised_L * 255.0).clip(0, 255).astype(np.uint8)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if not len(faces):
            return denoised_L

        output = denoised_L.copy()
        for fx, fy, fw, fh in faces:
            try:
                _blend_face(output, denoised_L, original_L, params,
                            int(fx), int(fy), int(fw), int(fh), img_h, img_w)
            except Exception:
                logger.warning("Face region skipped: %s", traceback.format_exc())

        return output.astype(np.float32)

    except Exception:
        logger.error("face_aware.apply failed — returning unchanged: %s", traceback.format_exc())
        return denoised_L


def _blend_face(output, denoised_L, original_L, params,
                fx, fy, fw, fh, img_h, img_w) -> None:
    """Apply lighter wavelet denoising + stronger detail recovery to one face box.

    Modifies output in-place via a feathered Gaussian mask.
    """
    # Expand box 20 % on each side, clamp to image bounds
    px, py = int(fw * 0.20), int(fh * 0.20)
    x  = max(0, fx - px);  x2 = min(img_w, fx + fw + px)
    y  = max(0, fy - py);  y2 = min(img_h, fy + fh + py)
    w, h = x2 - x, y2 - y

    # Feathered mask: 1.0 inside face box, tapering to 0 at edges
    mask = np.zeros((img_h, img_w), dtype=np.float64)
    mask[y:y2, x:x2] = 1.0
    mask = scipy.ndimage.gaussian_filter(mask, sigma=min(h, w) * 0.15)
    m_max = mask.max()
    if m_max > 0:
        mask /= m_max

    face_patch = denoised_L[y:y2, x:x2]   # float32 (h, w)
    orig_patch = original_L[y:y2, x:x2]   # float32 (h, w)

    # Lighter denoising pass for the face region
    face_sigma = float(params.get("wavelet_sigma", 0.04)) * float(params["face_strength_ratio"])

    if _SKIMAGE_AVAILABLE and min(h, w) >= 16:
        # Limit wavelet levels so they fit within the patch dimensions
        max_levels = max(1, int(np.floor(np.log2(min(h, w)))) - 2)
        levels = min(3, max_levels)
        face_denoised = _skrest.denoise_wavelet(
            face_patch,
            sigma=max(0.001, face_sigma),
            wavelet="db4",
            mode="soft",
            wavelet_levels=levels,
            method="BayesShrink",
            rescale_sigma=True,
            channel_axis=None,
        ).astype(np.float32)
    else:
        # Gaussian fallback for small patches or missing scikit-image
        sigma_px = max(0.5, face_sigma * 255.0 * 0.15)
        face_denoised = scipy.ndimage.gaussian_filter(face_patch, sigma=sigma_px).astype(np.float32)

    # Stronger detail recovery for the face (1.3 × background multiplier)
    blurred        = scipy.ndimage.gaussian_filter(orig_patch.astype(np.float64), sigma=1.5)
    face_detail    = orig_patch.astype(np.float64) - blurred
    face_local_var = scipy.ndimage.uniform_filter(face_detail ** 2, size=5)

    if _SKIMAGE_AVAILABLE:
        face_noise_sigma = float(_skrest.estimate_sigma(orig_patch))
    else:
        face_noise_sigma = float(np.std(orig_patch)) * 0.1

    face_dmask  = (face_local_var > face_noise_sigma ** 2).astype(np.float64)
    face_dmask  = scipy.ndimage.gaussian_filter(face_dmask, sigma=2.0)

    detail_str   = float(params["detail_strength"])
    face_recovered = face_denoised + (face_detail * face_dmask * detail_str * 1.3).astype(np.float32)
    face_recovered = np.clip(face_recovered, 0.0, 1.0).astype(np.float32)

    # Feathered blend into output
    mp = mask[y:y2, x:x2].astype(np.float32)
    output[y:y2, x:x2] = face_recovered * mp + output[y:y2, x:x2] * (1.0 - mp)
