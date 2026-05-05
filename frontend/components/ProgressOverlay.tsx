"use client";

import React from "react";

interface ProgressOverlayProps {
  isVisible: boolean;
}

/** Full-screen overlay shown while the full image is being denoised. */
export function ProgressOverlay({ isVisible }: ProgressOverlayProps) {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/85 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-6 text-center px-8">
        <div className="relative h-16 w-16">
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-[#2a2a2a] border-t-blue-500" />
        </div>
        <div>
          <p className="text-lg font-semibold text-white">Denoising full image</p>
          <p className="mt-2 text-sm text-[#888]">This may take 15–30 seconds depending on file size</p>
        </div>
        <p className="text-xs text-[#555]">Non-Local Means processing in progress…</p>
      </div>
    </div>
  );
}
