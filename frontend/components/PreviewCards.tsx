"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import type { AnalyzedZone } from "@/lib/api";

const ZONE_LABELS: Record<string, string> = {
  highlights: "Highlights",
  shadows: "Shadows / Blacks",
  faces: "Faces / People",
  center: "Center",
};

interface PreviewCardProps {
  zone: AnalyzedZone;
  denoisedBase64: string | null;
  isLoading: boolean;
}

/** Single preview card showing original vs denoised crop with hover toggle. */
function PreviewCard({ zone, denoisedBase64, isLoading }: PreviewCardProps) {
  const [showDenoised, setShowDenoised] = useState(false);

  const activeBase64 = showDenoised && denoisedBase64 ? denoisedBase64 : zone.thumbnail_base64;
  const label = ZONE_LABELS[zone.label] ?? zone.label;

  return (
    <div className="relative flex flex-col gap-2 rounded-xl border border-[#2a2a2a] bg-[#141414] overflow-hidden">
      {/* Image area */}
      <div
        className="relative aspect-square w-full overflow-hidden bg-[#0a0a0a]"
        onMouseEnter={() => denoisedBase64 && setShowDenoised(true)}
        onMouseLeave={() => setShowDenoised(false)}
        onTouchStart={() => denoisedBase64 && setShowDenoised((v) => !v)}
      >
        {/* Loading skeleton */}
        {isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#0a0a0a]/80">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#2a2a2a] border-t-blue-500" />
          </div>
        )}

        {/* Image */}
        <img
          src={`data:image/jpeg;base64,${activeBase64}`}
          alt={label}
          className="h-full w-full object-cover transition-opacity duration-200"
          draggable={false}
        />

        {/* Original / Denoised badge */}
        <div className="absolute bottom-2 left-2">
          <span
            className={cn(
              "rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
              showDenoised && denoisedBase64
                ? "bg-blue-500/80 text-white"
                : "bg-black/60 text-[#aaa]",
            )}
          >
            {showDenoised && denoisedBase64 ? "Denoised" : "Original"}
          </span>
        </div>

        {/* "Preview unavailable" fallback */}
        {!isLoading && !denoisedBase64 && (
          <div className="absolute bottom-2 right-2">
            <span className="rounded px-1.5 py-0.5 text-[10px] bg-black/60 text-[#666]">
              Preview unavailable
            </span>
          </div>
        )}
      </div>

      {/* Zone label */}
      <div className="px-3 pb-3">
        <p className="text-xs font-medium text-[#888]">{label}</p>
        {denoisedBase64 && (
          <p className="text-[10px] text-[#555] mt-0.5">
            {showDenoised ? "Tap/hover for original" : "Tap/hover for denoised"}
          </p>
        )}
      </div>
    </div>
  );
}

interface PreviewCardsProps {
  zones: AnalyzedZone[];
  denoisedPreviews: (string | null)[];
  isLoadingPreviews: boolean[];
}

/** Three-column grid of smart crop preview cards. */
export function PreviewCards({ zones, denoisedPreviews, isLoadingPreviews }: PreviewCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {zones.map((zone, i) => (
        <PreviewCard
          key={zone.label + i}
          zone={zone}
          denoisedBase64={denoisedPreviews[i] ?? null}
          isLoading={isLoadingPreviews[i] ?? false}
        />
      ))}
    </div>
  );
}
