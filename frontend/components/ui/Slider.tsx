"use client";

import * as RadixSlider from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

/** Styled slider with large touch target for iPad. */
export function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
}: SliderProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-white">{label}</span>
        <span className="text-sm tabular-nums text-[#888]">{value}</span>
      </div>
      <RadixSlider.Root
        className={cn(
          "relative flex items-center select-none touch-none",
          "h-11 w-full cursor-pointer",
          disabled && "opacity-40 pointer-events-none",
        )}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        disabled={disabled}
      >
        <RadixSlider.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-[#2a2a2a]">
          <RadixSlider.Range className="absolute h-full bg-blue-500" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          className={cn(
            "block h-5 w-5 rounded-full border-2 border-blue-500 bg-white shadow-lg",
            "ring-offset-background transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2",
          )}
          aria-label={label}
        />
      </RadixSlider.Root>
    </div>
  );
}
