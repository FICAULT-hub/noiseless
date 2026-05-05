"use client";

import React, { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/ToastProvider";

const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".avif", ".jxl"];
const REJECTED_EXTENSIONS = [".dng", ".raw", ".heic", ".heif", ".cr2", ".nef", ".arw"];
const MAX_BYTES = 60 * 1024 * 1024;

const FORMAT_LABELS: Record<string, string> = {
  ".jpg": "JPEG",
  ".jpeg": "JPEG",
  ".tif": "TIFF",
  ".tiff": "TIFF",
  ".avif": "AVIF",
  ".jxl": "JXL",
};

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  isLoading: boolean;
}

/** Drag-and-drop / tap-to-browse upload zone. */
export function UploadZone({ onFileSelected, isLoading }: UploadZoneProps) {
  const { toast } = useToast();
  const [isDragging, setIsDragging] = useState(false);
  const [droppedFile, setDroppedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback(
    (file: File): boolean => {
      const ext = "." + file.name.split(".").pop()!.toLowerCase();
      if (REJECTED_EXTENSIONS.includes(ext)) {
        toast({
          title: "Unsupported format",
          description: `${ext.toUpperCase()} is not supported. Please export as JPEG or TIFF from Lightroom Mobile.`,
          variant: "destructive",
        });
        return false;
      }
      if (!ACCEPTED_EXTENSIONS.includes(ext) && !file.type.startsWith("image/")) {
        toast({
          title: "Unsupported file",
          description: "Accepted formats: JPEG, TIFF, AVIF, JXL.",
          variant: "destructive",
        });
        return false;
      }
      if (file.size > MAX_BYTES) {
        toast({
          title: "File too large",
          description: `Maximum file size is 60 MB. Your file is ${formatBytes(file.size)}.`,
          variant: "destructive",
        });
        return false;
      }
      return true;
    },
    [toast],
  );

  const handleFile = useCallback(
    (file: File) => {
      if (!validate(file)) return;
      setDroppedFile(file);
      onFileSelected(file);
    },
    [validate, onFileSelected],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      e.target.value = "";
    },
    [handleFile],
  );

  const ext = droppedFile ? "." + droppedFile.name.split(".").pop()!.toLowerCase() : null;
  const formatLabel = ext ? (FORMAT_LABELS[ext] ?? ext.slice(1).toUpperCase()) : null;

  return (
    <div
      className={cn(
        "relative flex flex-col items-center justify-center",
        "min-h-[280px] w-full max-w-2xl mx-auto rounded-2xl border-2 border-dashed",
        "transition-colors duration-200 cursor-pointer",
        isDragging
          ? "border-blue-500 bg-blue-500/5"
          : "border-[#2a2a2a] bg-[#141414] hover:border-[#3a3a3a]",
        isLoading && "pointer-events-none opacity-60",
      )}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload image"
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={[...ACCEPTED_EXTENSIONS, "image/avif", "image/tiff", "image/jpeg"].join(",")}
        onChange={onInputChange}
      />

      {isLoading ? (
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#2a2a2a] border-t-blue-500" />
          <p className="text-sm text-[#888]">Analyzing image…</p>
        </div>
      ) : droppedFile ? (
        <div className="flex flex-col items-center gap-3 text-center px-6">
          <div className="flex items-center gap-2">
            {formatLabel && (
              <span className="rounded px-2 py-0.5 text-xs font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30">
                {formatLabel}
              </span>
            )}
            <span className="text-sm text-[#888]">{formatBytes(droppedFile.size)}</span>
          </div>
          <p className="text-sm font-medium text-white truncate max-w-xs">{droppedFile.name}</p>
          <p className="text-xs text-[#555]">Click or drop to replace</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4 text-center px-6">
          <svg
            className="h-12 w-12 text-[#333]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <div>
            <p className="text-base font-medium text-white">Drop your photo here</p>
            <p className="mt-1 text-sm text-[#888]">or tap to browse</p>
          </div>
          <div className="flex flex-wrap justify-center gap-1.5">
            {["JPEG", "TIFF", "AVIF", "JXL"].map((fmt) => (
              <span
                key={fmt}
                className="rounded px-2 py-0.5 text-xs text-[#555] border border-[#2a2a2a]"
              >
                {fmt}
              </span>
            ))}
          </div>
          <p className="text-xs text-[#444]">Max 60 MB</p>
        </div>
      )}
    </div>
  );
}
