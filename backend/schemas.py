"""Pydantic schemas for Denoise Studio API."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional


class CropRegion(BaseModel):
    """A rectangular crop region within an image."""
    x: int = Field(..., ge=0, description="Left edge pixel coordinate")
    y: int = Field(..., ge=0, description="Top edge pixel coordinate")
    w: int = Field(..., gt=0, description="Width in pixels")
    h: int = Field(..., gt=0, description="Height in pixels")


class AnalyzedZone(BaseModel):
    """A detected zone with its crop coordinates and thumbnail."""
    x: int
    y: int
    w: int
    h: int
    label: str = Field(..., description="highlights | shadows | faces | center")
    thumbnail_base64: str = Field(..., description="Base64-encoded JPEG of the original crop")


class AnalyzeResponse(BaseModel):
    """Response from the /analyze endpoint."""
    regions: List[AnalyzedZone]
    image_width: int
    image_height: int


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class BatchDenoiseResult(BaseModel):
    """Result for a single file in a batch denoise operation."""
    filename: str
    success: bool
    error: Optional[str] = None
