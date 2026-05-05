"use client";

import React from "react";
import { Slider } from "@/components/ui/Slider";

interface DenoiseSlidersProps {
  luminance: number;
  color: number;
  onLuminanceChange: (v: number) => void;
  onColorChange: (v: number) => void;
  disabled?: boolean;
}

/** Luminance and color noise strength sliders. */
export function DenoiseSliders({
  luminance,
  color,
  onLuminanceChange,
  onColorChange,
  disabled = false,
}: DenoiseSlidersProps) {
  return (
    <div className="rounded-xl border border-[#2a2a2a] bg-[#141414] p-5 flex flex-col gap-5">
      <h2 className="text-sm font-semibold text-[#888] uppercase tracking-wider">Noise Reduction</h2>
      <Slider
        label="Luminance"
        value={luminance}
        onChange={onLuminanceChange}
        disabled={disabled}
      />
      <Slider
        label="Color"
        value={color}
        onChange={onColorChange}
        disabled={disabled}
      />
    </div>
  );
}
