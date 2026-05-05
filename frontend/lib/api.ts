/**
 * API client for Denoise Studio backend.
 * All requests go to NEXT_PUBLIC_API_URL.
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface CropRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface AnalyzedZone extends CropRegion {
  label: "highlights" | "shadows" | "faces" | "center";
  thumbnail_base64: string;
}

export interface AnalyzeResponse {
  regions: AnalyzedZone[];
  image_width: number;
  image_height: number;
}

export interface ApiError {
  error: string;
  message: string;
}

/** Health check. Returns true if the backend is reachable. */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Analyze an image and return smart crop zones (highlights, shadows, faces/center).
 * Throws ApiError on backend error or network failure.
 */
export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ error: "NETWORK_ERROR", message: "Could not reach the server." }));
    throw detail.detail ?? detail;
  }
  return res.json() as Promise<AnalyzeResponse>;
}

/**
 * Fetch denoised preview thumbnails for a set of crop regions.
 * Returns an array of base64 JPEG strings, one per region.
 */
export async function fetchPreviews(
  file: File,
  regions: CropRegion[],
  luminanceStrength: number,
  colorStrength: number,
  signal?: AbortSignal,
): Promise<string[]> {
  const form = new FormData();
  form.append("file", file);
  form.append("luminance_strength", String(luminanceStrength));
  form.append("color_strength", String(colorStrength));
  form.append("output_format", "jpeg");
  form.append("preview_mode", "true");
  form.append("preview_regions", JSON.stringify(regions));

  const res = await fetch(`${BASE_URL}/denoise`, { method: "POST", body: form, signal });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ error: "NETWORK_ERROR", message: "Could not reach the server." }));
    throw detail.detail ?? detail;
  }
  const json = (await res.json()) as { previews: string[] };
  return json.previews;
}

/**
 * Denoise the full image and return a Blob for download.
 *
 * @param file - The original uploaded file
 * @param luminanceStrength - 0.0–1.0 slider value
 * @param colorStrength - 0.0–1.0 slider value
 * @param outputFormat - "jpeg" | "tiff"
 * @returns Tuple of [blob, suggestedFilename]
 */
export async function denoiseFullImage(
  file: File,
  luminanceStrength: number,
  colorStrength: number,
  outputFormat: "jpeg" | "tiff",
): Promise<[Blob, string]> {
  const form = new FormData();
  form.append("file", file);
  form.append("luminance_strength", String(luminanceStrength));
  form.append("color_strength", String(colorStrength));
  form.append("output_format", outputFormat);
  form.append("preview_mode", "false");

  const res = await fetch(`${BASE_URL}/denoise`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ error: "NETWORK_ERROR", message: "Could not reach the server." }));
    throw detail.detail ?? detail;
  }

  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="([^"]+)"/);
  const ext = outputFormat === "tiff" ? ".tiff" : ".jpg";
  const stem = file.name.replace(/\.[^.]+$/, "");
  const filename = match ? match[1] : `${stem}_denoised${ext}`;
  return [blob, filename];
}

/** Trigger a browser file download from a Blob. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
