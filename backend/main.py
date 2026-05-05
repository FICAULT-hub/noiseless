"""Denoise Studio — FastAPI backend.

Endpoints:
  POST /analyze        — detect smart crop zones in an uploaded image
  POST /denoise        — full or preview-mode NLM denoising
  POST /denoise-batch  — batch denoise (scaffold, not yet exposed in UI)
  GET  /health         — health check
"""

import io
import json
import logging
import os
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image

# Optional format plugins — import side-effects register themselves
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass
try:
    import imagecodecs  # noqa: F401
except ImportError:
    pass

from analyze import detect_zones
from denoise import denoise_array, denoise_crops, encode_crop_jpeg, encode_image
from schemas import AnalyzeResponse, CropRegion, ErrorResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed origins
# ---------------------------------------------------------------------------
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
origins = ["*"] if ALLOWED_ORIGIN == "*" else [ALLOWED_ORIGIN]

app = FastAPI(title="Denoise Studio API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for NLM processing
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKER_THREADS", "2")))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_BYTES = 60 * 1024 * 1024  # 60 MB
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".avif", ".jxl"}
UNSUPPORTED_EXTENSIONS = {".dng", ".raw", ".cr2", ".nef", ".arw", ".raf", ".heic", ".heif"}
UNSUPPORTED_MESSAGES = {
    ".dng": "DNG files are not supported. Please export as JPEG or TIFF from Lightroom Mobile.",
    ".raw": "RAW files are not supported. Please export as JPEG or TIFF from Lightroom Mobile.",
    ".heic": "HEIC files are not supported. Please export as JPEG from Lightroom Mobile.",
    ".heif": "HEIF files are not supported. Please export as JPEG from Lightroom Mobile.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile, data: bytes) -> None:
    """Raise HTTPException if the file is unsupported or too large."""
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "FILE_TOO_LARGE", "message": "File exceeds the 60 MB limit. Please reduce file size before uploading."},
        )
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext in UNSUPPORTED_EXTENSIONS:
        msg = UNSUPPORTED_MESSAGES.get(ext, f"{ext.upper()} files are not supported. Please export as JPEG or TIFF.")
        raise HTTPException(
            status_code=415,
            detail={"error": "UNSUPPORTED_FORMAT", "message": msg},
        )
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={"error": "UNSUPPORTED_FORMAT", "message": f"Unsupported file format '{ext}'. Accepted formats: JPEG, TIFF, AVIF, JXL."},
        )


def _load_image_bgr(data: bytes, filename: str = "") -> np.ndarray:
    """Decode image bytes to a BGR uint8 OpenCV array.

    Handles JPEG, TIFF via OpenCV, and AVIF/JXL via Pillow plugins.
    """
    ext = os.path.splitext(filename)[1].lower()

    # Try OpenCV first (fast path for JPEG/TIFF/PNG)
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr

    # Fallback to Pillow (handles AVIF/JXL with plugins)
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.error("Failed to decode image %s: %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail={"error": "DECODE_ERROR", "message": "Could not decode the uploaded image. The file may be corrupt or in an unsupported variant."},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Detect highlights, shadows, and face/center zones in the uploaded image.

    Returns three AnalyzedZone objects each with a base64 JPEG thumbnail
    of the original (un-denoised) crop region.
    """
    try:
        data = await file.read()
        _validate_upload(file, data)
        bgr = _load_image_bgr(data, file.filename or "")
        img_h, img_w = bgr.shape[:2]
        zones = detect_zones(bgr)
        return AnalyzeResponse(regions=zones, image_width=img_w, image_height=img_h)
    except HTTPException:
        raise
    except MemoryError:
        logger.error("OOM during /analyze: %s", traceback.format_exc())
        raise HTTPException(
            status_code=413,
            detail={"error": "OUT_OF_MEMORY", "message": "Image is too large to process. Try a smaller file."},
        )
    except Exception:
        logger.error("Unexpected error in /analyze: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred during analysis."},
        )


@app.post("/denoise")
async def denoise(
    file: UploadFile = File(...),
    luminance_strength: float = Form(0.4),
    color_strength: float = Form(0.3),
    output_format: str = Form("jpeg"),
    preview_mode: bool = Form(False),
    preview_regions: Optional[str] = Form(None),
) -> Response:
    """Denoise an image using OpenCV Non-Local Means.

    In preview_mode=true, only the specified preview_regions are denoised
    and returned as a JSON array of base64-encoded JPEG strings.

    In preview_mode=false (default), the full image is denoised and returned
    as a binary image file (JPEG or TIFF).
    """
    try:
        data = await file.read()
        _validate_upload(file, data)

        luminance_strength = max(0.0, min(1.0, luminance_strength))
        color_strength = max(0.0, min(1.0, color_strength))

        if output_format not in ("jpeg", "tiff"):
            output_format = "jpeg"

        bgr = _load_image_bgr(data, file.filename or "")

        if preview_mode:
            regions: List[CropRegion] = []
            if preview_regions:
                try:
                    raw_regions = json.loads(preview_regions)
                    regions = [CropRegion(**r) for r in raw_regions]
                except Exception:
                    raise HTTPException(
                        status_code=422,
                        detail={"error": "INVALID_REGIONS", "message": "preview_regions must be a valid JSON array of {x,y,w,h} objects."},
                    )

            import asyncio
            loop = asyncio.get_event_loop()
            denoised_crops = await loop.run_in_executor(
                _executor,
                lambda: denoise_crops(bgr, regions, luminance_strength, color_strength),
            )
            thumbnails = [encode_crop_jpeg(c) for c in denoised_crops]
            return Response(
                content=json.dumps({"previews": thumbnails}),
                media_type="application/json",
            )

        # Full image denoise
        import asyncio
        loop = asyncio.get_event_loop()
        denoised = await loop.run_in_executor(
            _executor,
            lambda: denoise_array(bgr, luminance_strength, color_strength),
        )
        image_bytes, media_type = encode_image(denoised, output_format)
        ext = ".tiff" if output_format == "tiff" else ".jpg"
        original_stem = os.path.splitext(file.filename or "image")[0]
        disposition = f'attachment; filename="{original_stem}_denoised{ext}"'

        return Response(
            content=image_bytes,
            media_type=media_type,
            headers={"Content-Disposition": disposition},
        )

    except HTTPException:
        raise
    except MemoryError:
        logger.error("OOM during /denoise: %s", traceback.format_exc())
        raise HTTPException(
            status_code=413,
            detail={"error": "OUT_OF_MEMORY", "message": "Image is too large to process. Try a smaller file."},
        )
    except Exception:
        logger.error("Unexpected error in /denoise: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred during denoising."},
        )


@app.post("/denoise-batch")
async def denoise_batch(
    files: List[UploadFile] = File(...),
    luminance_strength: float = Form(0.4),
    color_strength: float = Form(0.3),
    output_format: str = Form("jpeg"),
) -> StreamingResponse:
    """Batch denoise up to 10 images and return them as a ZIP archive.

    Processes files sequentially to avoid memory pressure.
    NOTE: This endpoint is scaffolded but not yet exposed in the UI (BATCH_ENABLED=false).
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail={"error": "TOO_MANY_FILES", "message": "Batch mode supports a maximum of 10 files at once."},
        )

    ext = ".tiff" if output_format == "tiff" else ".jpg"

    async def generate_zip():
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for upload in files:
                try:
                    data = await upload.read()
                    _validate_upload(upload, data)
                    bgr = _load_image_bgr(data, upload.filename or "")

                    import asyncio
                    loop = asyncio.get_event_loop()
                    denoised = await loop.run_in_executor(
                        _executor,
                        lambda b=bgr: denoise_array(b, luminance_strength, color_strength),
                    )
                    img_bytes, _ = encode_image(denoised, output_format)
                    stem = os.path.splitext(upload.filename or "image")[0]
                    zf.writestr(f"{stem}_denoised{ext}", img_bytes)
                except Exception as exc:
                    logger.error("Batch error for %s: %s", upload.filename, exc)
                    zf.writestr(f"{upload.filename}.error.txt", str(exc))

        zip_buf.seek(0)
        yield zip_buf.read()

    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="denoised_batch.zip"'},
    )
