"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { LoupeController } from "@/components/Loupe";
import { cn } from "@/lib/utils";

interface BeforeAfterViewerProps {
  originalSrc: string;
  denoisedSrc: string;
  imageWidth: number;
  imageHeight: number;
  onReset: () => void;
}

/**
 * Full-width before/after comparison with a draggable vertical split slider
 * and a circular loupe for pixel-level comparison.
 */
export function BeforeAfterViewer({
  originalSrc,
  denoisedSrc,
  imageWidth,
  imageHeight,
  onReset,
}: BeforeAfterViewerProps) {
  const [splitRatio, setSplitRatio] = useState(0.5);
  const [isDragging, setIsDragging] = useState(false);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const clamp = (v: number) => Math.max(0.02, Math.min(0.98, v));

  const updateSplit = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    setSplitRatio(clamp((clientX - rect.left) / rect.width));
  }, []);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
    },
    [],
  );

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      updateSplit(e.clientX);
    },
    [isDragging, updateSplit],
  );

  const onMouseUp = useCallback(() => setIsDragging(false), []);

  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  const onTouchMove = useCallback(
    (e: React.TouchEvent) => {
      updateSplit(e.touches[0].clientX);
    },
    [updateSplit],
  );

  const splitPct = `${(splitRatio * 100).toFixed(1)}%`;

  return (
    <div className="w-full flex flex-col gap-6">
      <LoupeController
        originalSrc={originalSrc}
        denoisedSrc={denoisedSrc}
        imageWidth={imageWidth}
        imageHeight={imageHeight}
      >
        <div
          ref={containerRef}
          className={cn(
            "relative w-full overflow-hidden rounded-xl border border-[#2a2a2a] bg-[#0a0a0a]",
            "select-none touch-none",
          )}
          style={{ maxHeight: "80vh" }}
        >
          {/* Denoised (bottom layer, full width) */}
          <img
            src={denoisedSrc}
            alt="Denoised"
            className="w-full block object-contain"
            draggable={false}
          />

          {/* Original (top layer, clipped to left of split) */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ width: splitPct }}
          >
            <img
              src={originalSrc}
              alt="Original"
              className="block object-contain"
              style={{ width: containerWidth > 0 ? containerWidth : "100%", maxWidth: "none" }}
              draggable={false}
            />
          </div>

          {/* Split handle */}
          <div
            className={cn(
              "absolute inset-y-0 z-10 flex items-center justify-center",
              "cursor-ew-resize touch-none",
            )}
            style={{ left: splitPct, transform: "translateX(-50%)" }}
            onMouseDown={onMouseDown}
            onTouchMove={onTouchMove}
          >
            <div className="h-full w-0.5 bg-white/40" />
            <div className="absolute flex h-9 w-9 items-center justify-center rounded-full bg-white shadow-lg">
              <svg className="h-4 w-4 text-black" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" d="M8 12H4m0 0l3-3m-3 3l3 3M16 12h4m0 0l-3-3m3 3l-3 3" />
              </svg>
            </div>
          </div>

          {/* Corner labels */}
          <div className="pointer-events-none absolute left-3 top-3 rounded px-2 py-0.5 text-xs font-semibold bg-black/60 text-white/80">
            Original
          </div>
          <div className="pointer-events-none absolute right-3 top-3 rounded px-2 py-0.5 text-xs font-semibold bg-blue-600/80 text-white">
            Denoised
          </div>
        </div>
      </LoupeController>

      {/* Loupe hint */}
      <p className="text-center text-xs text-[#555]">
        Drag the divider to compare · Move cursor or finger over the image to use the loupe
      </p>

      {/* Reset button */}
      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="rounded-lg border border-[#2a2a2a] bg-[#141414] px-6 py-2.5 text-sm font-medium text-white hover:border-[#3a3a3a] hover:bg-[#1e1e1e] transition-colors"
        >
          Process Another Photo
        </button>
      </div>
    </div>
  );
}
