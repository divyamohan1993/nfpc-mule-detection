"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Image from "next/image";

interface LightboxImage {
  src: string;
  alt: string;
}

export function LightboxGallery({
  images,
}: {
  images: LightboxImage[];
}) {
  const [open, setOpen] = useState(false);
  const [idx, setIdx] = useState(0);
  const touchRef = useRef(0);

  const close = useCallback(() => setOpen(false), []);
  const prev = useCallback(() => setIdx((i) => (i > 0 ? i - 1 : images.length - 1)), [images.length]);
  const next = useCallback(() => setIdx((i) => (i < images.length - 1 ? i + 1 : 0)), [images.length]);

  useEffect(() => {
    if (!open) return;
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      if (e.key === "ArrowLeft") prev();
      if (e.key === "ArrowRight") next();
    };
    const handleTouchStart = (e: TouchEvent) => { touchRef.current = e.touches[0].clientX; };
    const handleTouchEnd = (e: TouchEvent) => {
      const dx = e.changedTouches[0].clientX - touchRef.current;
      if (Math.abs(dx) > 50) dx < 0 ? next() : prev();
    };
    document.addEventListener("keydown", handle);
    document.addEventListener("touchstart", handleTouchStart, { passive: true });
    document.addEventListener("touchend", handleTouchEnd);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handle);
      document.removeEventListener("touchstart", handleTouchStart);
      document.removeEventListener("touchend", handleTouchEnd);
      document.body.style.overflow = "";
    };
  }, [open, close, prev, next]);

  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {images.map((img, i) => (
          <button
            key={img.src}
            type="button"
            onClick={() => { setIdx(i); setOpen(true); }}
            className="group cursor-zoom-in overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-raised)] text-left transition-all hover:border-[var(--color-accent)]/20"
          >
            <div className="relative aspect-[4/3] overflow-hidden bg-white">
              <Image
                src={img.src}
                alt={img.alt}
                fill
                className="object-contain p-2 transition-transform group-hover:scale-[1.02]"
                sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
                loading="lazy"
              />
            </div>
            <div className="border-t border-[var(--color-border)]/50 px-3 py-2.5 sm:px-4 sm:py-3">
              <span className="text-xs font-medium text-[#a0a0a0] sm:text-sm">{img.alt}</span>
            </div>
          </button>
        ))}
      </div>

      {open && (
        <div
          className="fixed inset-0 z-[200] flex items-center justify-center bg-black/90 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) close(); }}
          role="dialog"
          aria-modal="true"
          aria-label={images[idx].alt}
        >
          {/* Close */}
          <button
            onClick={close}
            className="absolute right-3 top-3 z-10 flex h-12 w-12 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20 sm:right-4 sm:top-4"
            aria-label="Close"
          >
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>

          {/* Prev */}
          <button
            onClick={prev}
            className="absolute left-2 top-1/2 z-10 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20 sm:left-4"
            aria-label="Previous image"
          >
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>

          {/* Next */}
          <button
            onClick={next}
            className="absolute right-2 top-1/2 z-10 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20 sm:right-4"
            aria-label="Next image"
          >
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </button>

          {/* Image */}
          <div className="relative mx-2 my-14 max-h-[85vh] max-w-[96vw] sm:mx-8 sm:my-16 sm:max-w-[90vw] lg:mx-16">
            <Image
              src={images[idx].src}
              alt={images[idx].alt}
              width={1200}
              height={900}
              className="max-h-[85vh] w-auto rounded-lg object-contain"
              priority
            />
          </div>

          {/* Counter + label */}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-center sm:bottom-4">
            <div className="text-xs font-medium text-white sm:text-sm">{images[idx].alt}</div>
            <div className="mt-1 text-xs text-white/40 tabular-nums">
              {idx + 1} / {images.length}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
