"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";

const LOUPE_SIZE = 200;
const MAGNIFICATION = 3;

interface LoupeProps {
  /** URL of the original image (left side of split). */
  originalSrc: string;
  /** URL of the denoised image (right side of split). */
  denoisedSrc: string;
  /** Natural width of the images in pixels. */
  imageWidth: number;
  /** Natural height of the images in pixels. */
  imageHeight: number;
  /** Rendered width of the container element in pixels. */
  containerWidth: number;
  /** Rendered height of the container element in pixels. */
  containerHeight: number;
  /** Current X position of the loupe centre (relative to container). */
  posX: number;
  /** Current Y position of the loupe centre (relative to container). */
  posY: number;
}

/**
 * Circular loupe that shows a 3× magnified split view of original vs denoised.
 * The left half of the loupe shows the original; the right half shows denoised.
 */
export function Loupe({
  originalSrc,
  denoisedSrc,
  imageWidth,
  imageHeight,
  containerWidth,
  containerHeight,
  posX,
  posY,
}: LoupeProps) {
  const half = LOUPE_SIZE / 2;
  // Scale from container pixels to image pixels
  const scaleX = imageWidth / containerWidth;
  const scaleY = imageHeight / containerHeight;

  // The center of the loupe in image pixel space
  const imgCX = posX * scaleX;
  const imgCY = posY * scaleY;

  // Displayed image size inside the loupe: LOUPE_SIZE × LOUPE_SIZE shows a
  // region of (LOUPE_SIZE / MAGNIFICATION) image pixels on each side.
  const regionPx = LOUPE_SIZE / MAGNIFICATION;

  // Background offset so the region is centred: we want the image at
  // -(imgCX - regionPx/2) × MAGNIFICATION from the left of the container.
  const bgOffX = -(imgCX - regionPx / 2) * MAGNIFICATION;
  const bgOffY = -(imgCY - regionPx / 2) * MAGNIFICATION;
  const bgSize = `${imageWidth * MAGNIFICATION}px ${imageHeight * MAGNIFICATION}px`;

  // Position the loupe so it doesn't overflow the container edges
  const loupeLeft = Math.max(half, Math.min(containerWidth - half, posX)) - half;
  const loupeTop = Math.max(half, Math.min(containerHeight - half, posY)) - half;

  return (
    <div
      className="pointer-events-none absolute z-20 overflow-hidden rounded-full border-2 border-white/30 shadow-2xl"
      style={{
        width: LOUPE_SIZE,
        height: LOUPE_SIZE,
        left: loupeLeft,
        top: loupeTop,
      }}
    >
      {/* Left half — original */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `url(${originalSrc})`,
          backgroundRepeat: "no-repeat",
          backgroundSize: bgSize,
          backgroundPosition: `${bgOffX}px ${bgOffY}px`,
          clipPath: `inset(0 50% 0 0)`,
        }}
      />
      {/* Right half — denoised */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `url(${denoisedSrc})`,
          backgroundRepeat: "no-repeat",
          backgroundSize: bgSize,
          backgroundPosition: `${bgOffX}px ${bgOffY}px`,
          clipPath: `inset(0 0 0 50%)`,
        }}
      />
      {/* Centre divider */}
      <div className="absolute inset-y-0 left-1/2 w-px bg-white/60" />
      {/* Labels */}
      <div className="absolute bottom-3 left-2 text-[8px] font-semibold text-white/70 uppercase tracking-widest">
        Orig
      </div>
      <div className="absolute bottom-3 right-2 text-[8px] font-semibold text-white/70 uppercase tracking-widest">
        New
      </div>
    </div>
  );
}

interface LoupeControllerProps {
  originalSrc: string;
  denoisedSrc: string;
  imageWidth: number;
  imageHeight: number;
  children: React.ReactNode;
}

/**
 * Wraps an image container and tracks pointer/touch position to drive the Loupe.
 * The loupe is rendered as an overlay inside the container.
 */
export function LoupeController({
  originalSrc,
  denoisedSrc,
  imageWidth,
  imageHeight,
  children,
}: LoupeControllerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const [dims, setDims] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDims({ w: width, h: height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const getRelativePos = useCallback((clientX: number, clientY: number) => {
    const el = containerRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }, []);

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      setPos(getRelativePos(e.clientX, e.clientY));
    },
    [getRelativePos],
  );

  const onTouchMove = useCallback(
    (e: React.TouchEvent) => {
      e.preventDefault();
      const t = e.touches[0];
      setPos(getRelativePos(t.clientX, t.clientY));
    },
    [getRelativePos],
  );

  return (
    <div
      ref={containerRef}
      className="relative select-none"
      onMouseMove={onMouseMove}
      onMouseLeave={() => setPos(null)}
      onTouchMove={onTouchMove}
      onTouchEnd={() => setPos(null)}
    >
      {children}
      {pos && dims.w > 0 && (
        <Loupe
          originalSrc={originalSrc}
          denoisedSrc={denoisedSrc}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          containerWidth={dims.w}
          containerHeight={dims.h}
          posX={pos.x}
          posY={pos.y}
        />
      )}
    </div>
  );
}
