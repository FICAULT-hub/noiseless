"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { UploadZone } from "@/components/UploadZone";
import { PreviewCards } from "@/components/PreviewCards";
import { DenoiseSliders } from "@/components/DenoiseSliders";
import { ProgressOverlay } from "@/components/ProgressOverlay";
import { BeforeAfterViewer } from "@/components/BeforeAfterViewer";
import { useToast } from "@/components/ui/ToastProvider";
import {
  analyzeImage,
  fetchPreviews,
  denoiseFullImage,
  downloadBlob,
  type AnalyzedZone,
  type CropRegion,
} from "@/lib/api";

/** Feature flag — batch mode is scaffolded but hidden until enabled. */
const BATCH_ENABLED = false;

type AppStep = "upload" | "configure" | "result";

interface ResultState {
  originalObjectUrl: string;
  denoisedObjectUrl: string;
  imageWidth: number;
  imageHeight: number;
}

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

export default function HomePage() {
  const { toast } = useToast();

  const [step, setStep] = useState<AppStep>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [zones, setZones] = useState<AnalyzedZone[]>([]);
  const [imageSize, setImageSize] = useState({ w: 0, h: 0 });

  // Slider state (0–100, converted to 0.0–1.0 for API)
  const [luminance, setLuminance] = useState(40);
  const [color, setColor] = useState(30);
  const debouncedLuminance = useDebounce(luminance, 400);
  const debouncedColor = useDebounce(color, 400);

  // Per-card preview state
  const [previews, setPreviews] = useState<(string | null)[]>([null, null, null]);
  const [previewsLoading, setPreviewsLoading] = useState<boolean[]>([false, false, false]);

  // Output format
  const [outputFormat, setOutputFormat] = useState<"jpeg" | "tiff">("jpeg");

  // Processing / result
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ResultState | null>(null);

  // AbortController ref for in-flight preview fetches
  const previewAbortRef = useRef<AbortController | null>(null);

  /** Fetch denoised previews for all detected zones. */
  const loadPreviews = useCallback(
    async (
      sourceFile: File,
      sourceZones: AnalyzedZone[],
      lumStrength: number,
      colStrength: number,
    ) => {
      if (sourceZones.length === 0) return;

      // Cancel previous in-flight request
      previewAbortRef.current?.abort();
      const controller = new AbortController();
      previewAbortRef.current = controller;

      setPreviewsLoading(sourceZones.map(() => true));

      const regions: CropRegion[] = sourceZones.map(({ x, y, w, h }) => ({ x, y, w, h }));

      try {
        const results = await fetchPreviews(
          sourceFile,
          regions,
          lumStrength,
          colStrength,
          controller.signal,
        );
        setPreviews(results.map((r) => r ?? null));
      } catch (err: unknown) {
        if ((err as { name?: string }).name === "AbortError") return;
        const apiErr = err as { message?: string };
        toast({
          title: "Preview unavailable",
          description: apiErr?.message ?? "Could not load previews — you can still denoise the full image.",
          variant: "destructive",
        });
        setPreviews(sourceZones.map(() => null));
      } finally {
        setPreviewsLoading(sourceZones.map(() => false));
      }
    },
    [toast],
  );

  /** Called when the user drops/selects a file. */
  const handleFileSelected = useCallback(
    async (selectedFile: File) => {
      setFile(selectedFile);
      setStep("upload");
      setZones([]);
      setPreviews([null, null, null]);
      setResult(null);
      setIsAnalyzing(true);

      try {
        const analysis = await analyzeImage(selectedFile);
        setZones(analysis.regions);
        setImageSize({ w: analysis.image_width, h: analysis.image_height });
        setStep("configure");
        await loadPreviews(selectedFile, analysis.regions, 0.4, 0.3);
      } catch (err: unknown) {
        const apiErr = err as { error?: string; message?: string };
        toast({
          title: "Analysis failed",
          description: apiErr?.message ?? "Could not analyze the image. You can still denoise without previews.",
          variant: "destructive",
        });
        setStep("configure");
      } finally {
        setIsAnalyzing(false);
      }
    },
    [toast, loadPreviews],
  );

  /** Re-fetch previews when debounced slider values change. */
  useEffect(() => {
    if (!file || zones.length === 0) return;
    loadPreviews(file, zones, debouncedLuminance / 100, debouncedColor / 100);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedLuminance, debouncedColor]);

  /** Denoise the full image and trigger download. */
  const handleDenoise = useCallback(async () => {
    if (!file) return;
    setIsProcessing(true);
    try {
      const [blob, filename] = await denoiseFullImage(
        file,
        luminance / 100,
        color / 100,
        outputFormat,
      );
      downloadBlob(blob, filename);

      // Build object URLs for before/after viewer
      const originalUrl = URL.createObjectURL(file);
      const denoisedUrl = URL.createObjectURL(blob);
      setResult({
        originalObjectUrl: originalUrl,
        denoisedObjectUrl: denoisedUrl,
        imageWidth: imageSize.w,
        imageHeight: imageSize.h,
      });
      setStep("result");
    } catch (err: unknown) {
      const apiErr = err as { error?: string; message?: string };
      let title = "Denoising failed";
      let description = apiErr?.message ?? "An unexpected error occurred.";
      if (apiErr?.error === "UNSUPPORTED_FORMAT") title = "Unsupported format";
      if (apiErr?.error === "FILE_TOO_LARGE") title = "File too large";
      if (apiErr?.error === "OUT_OF_MEMORY") {
        title = "Image too large";
        description = "The server ran out of memory. Try a smaller file.";
      }
      toast({ title, description, variant: "destructive" });
    } finally {
      setIsProcessing(false);
    }
  }, [file, luminance, color, outputFormat, imageSize, toast]);

  /** Reset the app to the upload state. */
  const handleReset = useCallback(() => {
    if (result) {
      URL.revokeObjectURL(result.originalObjectUrl);
      URL.revokeObjectURL(result.denoisedObjectUrl);
    }
    setFile(null);
    setZones([]);
    setPreviews([null, null, null]);
    setResult(null);
    setStep("upload");
  }, [result]);

  return (
    <main className="min-h-screen bg-[#0a0a0a] pb-16">
      <ProgressOverlay isVisible={isProcessing} />

      {/* Header */}
      <header className="border-b border-[#1a1a1a] px-6 py-4">
        <div className="mx-auto max-w-5xl flex items-center gap-3">
          <div className="h-6 w-6 rounded bg-blue-500 flex items-center justify-center">
            <svg className="h-3.5 w-3.5 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 3a9 9 0 100 18A9 9 0 0012 3zm0 2a7 7 0 110 14A7 7 0 0112 5zm0 2a5 5 0 100 10A5 5 0 0012 7z" />
            </svg>
          </div>
          <h1 className="text-base font-semibold text-white">Denoise Studio</h1>
          <span className="ml-auto text-xs text-[#444]">v1.0</span>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-6 pt-10 flex flex-col gap-8">
        {/* Step 1 — Upload */}
        <UploadZone onFileSelected={handleFileSelected} isLoading={isAnalyzing} />

        {/* Step 2+3 — Configure (shown after successful analysis) */}
        {step === "configure" && (
          <>
            {/* Smart Preview Cards */}
            {zones.length > 0 && (
              <section>
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-[#555]">
                  Smart Preview Zones
                </h2>
                <PreviewCards
                  zones={zones}
                  denoisedPreviews={previews}
                  isLoadingPreviews={previewsLoading}
                />
              </section>
            )}

            {/* Sliders */}
            <DenoiseSliders
              luminance={luminance}
              color={color}
              onLuminanceChange={setLuminance}
              onColorChange={setColor}
              disabled={isProcessing}
            />

            {/* Output format + Denoise button */}
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              {/* Format toggle */}
              <div className="flex items-center gap-3">
                <span className="text-sm text-[#888]">Output format</span>
                <div className="flex rounded-lg border border-[#2a2a2a] overflow-hidden">
                  {(["jpeg", "tiff"] as const).map((fmt) => (
                    <button
                      key={fmt}
                      onClick={() => setOutputFormat(fmt)}
                      className={`px-4 py-2 text-sm font-medium transition-colors ${
                        outputFormat === fmt
                          ? "bg-blue-600 text-white"
                          : "bg-[#141414] text-[#888] hover:text-white"
                      }`}
                    >
                      {fmt.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Denoise button */}
              <button
                onClick={handleDenoise}
                disabled={isProcessing || !file}
                className="rounded-xl bg-blue-600 px-8 py-3 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Denoise Full Image
              </button>
            </div>
          </>
        )}

        {/* Step 6 — Before/After result viewer */}
        {step === "result" && result && (
          <BeforeAfterViewer
            originalSrc={result.originalObjectUrl}
            denoisedSrc={result.denoisedObjectUrl}
            imageWidth={result.imageWidth}
            imageHeight={result.imageHeight}
            onReset={handleReset}
          />
        )}
      </div>

      {/* Batch mode — scaffolded but hidden */}
      {BATCH_ENABLED && (
        <div className="hidden">
          {/* Batch UI placeholder — enable BATCH_ENABLED to expose */}
        </div>
      )}
    </main>
  );
}
