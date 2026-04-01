import { useEffect, useRef, useState } from 'react';

export interface ScoredFrame {
  path: string;
  quality: 'good' | 'warn' | 'fail';
  sharpness: number;
  coverage: number;
  index: number;
}

interface PreviewImages {
  original: string;    // object URL (revoke when done)
  undistorted: string; // object URL (revoke when done)
}

interface Props {
  scoredFrames: ScoredFrame[];
  cameraMatrix: number[][];
  distCoeffs: number[];
  ws: WebSocket | null;
}

const QUALITY_DOT: Record<string, string> = {
  good: 'bg-emerald-400',
  warn: 'bg-yellow-400',
  fail: 'bg-red-400',
};

/** Convert a base64-encoded JPEG string to a short-lived object URL. */
function b64ToObjectUrl(b64: string): string {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
}

export default function UndistortPreview({ scoredFrames, cameraMatrix, distCoeffs, ws }: Props) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [images, setImages] = useState<PreviewImages | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [splitPos, setSplitPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const previewable = scoredFrames.filter((f) => f.quality !== 'fail');
  const selected = previewable[selectedIndex] ?? null;

  // Revoke object URLs when they are replaced or on unmount to free memory
  const prevImagesRef = useRef<PreviewImages | null>(null);
  useEffect(() => {
    const prev = prevImagesRef.current;
    if (prev) {
      URL.revokeObjectURL(prev.original);
      URL.revokeObjectURL(prev.undistorted);
    }
    prevImagesRef.current = images;
  }, [images]);
  useEffect(() => {
    return () => {
      if (prevImagesRef.current) {
        URL.revokeObjectURL(prevImagesRef.current.original);
        URL.revokeObjectURL(prevImagesRef.current.undistorted);
      }
    };
  }, []);

  // Listen for preview_result messages
  useEffect(() => {
    if (!ws) return;
    const handler = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.action !== 'preview_result') return;
        if (!msg.success) {
          setError(msg.error ?? 'Preview failed');
          setLoading(false);
          return;
        }
        // Convert base64 → object URLs so large JPEG strings are never held in
        // React state (avoids 5–10 MB allocations on every render).
        setImages({
          original:    b64ToObjectUrl(msg.original),
          undistorted: b64ToObjectUrl(msg.undistorted),
        });
        setLoading(false);
        setError(null);
      } catch {
        // ignore non-JSON
      }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  // Request preview whenever selection changes
  useEffect(() => {
    if (!selected || !ws || ws.readyState !== WebSocket.OPEN) return;
    setLoading(true);
    setImages(null);
    setError(null);
    ws.send(
      JSON.stringify({
        action: 'preview_undistort',
        path: selected.path,
        camera_matrix: cameraMatrix,
        dist_coeffs: distCoeffs,
      })
    );
  }, [selected?.path, ws]);

  // Drag-to-split on the image container
  const onPointerDown = (e: React.PointerEvent) => {
    isDragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updateSplit(e);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!isDragging.current) return;
    updateSplit(e);
  };
  const onPointerUp = () => { isDragging.current = false; };

  const updateSplit = (e: React.PointerEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100));
    setSplitPos(Math.round(pct));
  };

  if (previewable.length === 0) {
    return (
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-6 text-slate-400 text-sm">
        No previewable frames yet. Score frames first.
      </div>
    );
  }

  const filename = (p: string) => p.replace(/\\/g, '/').split('/').pop() ?? p;

  return (
    <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-4">
      <h2 className="text-sm font-semibold text-slate-300 tracking-wide uppercase">
        Undistort Preview
      </h2>

      {/* Frame selector */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-slate-400 shrink-0">Frame</label>
        <select
          value={selectedIndex}
          onChange={(e) => setSelectedIndex(Number(e.target.value))}
          className="flex-1 bg-slate-700 border border-slate-600 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
        >
          {previewable.map((f, i) => (
            <option key={f.path} value={i}>
              {filename(f.path)} — sharpness {f.sharpness.toFixed(0)}, coverage {(f.coverage * 100).toFixed(0)}%
            </option>
          ))}
        </select>
        {selected && (
          <span
            className={`w-2.5 h-2.5 rounded-full shrink-0 ${QUALITY_DOT[selected.quality]}`}
            title={selected.quality}
          />
        )}
      </div>

      {/* Image split view */}
      <div
        ref={containerRef}
        className="relative rounded-lg overflow-hidden bg-slate-900 select-none cursor-col-resize"
        style={{ aspectRatio: '16 / 9' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-20">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-20 text-red-400 text-xs px-4 text-center">
            {error}
          </div>
        )}

        {images && (
          <>
            {/* Original — bottom layer, always full width */}
            <img
              src={`data:image/jpeg;base64,${images.original}`}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              draggable={false}
            />

            {/* Undistorted — top layer, revealed from the left */}
            <img
              src={`data:image/jpeg;base64,${images.undistorted}`}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              style={{ clipPath: `inset(0 ${100 - splitPos}% 0 0)` }}
              draggable={false}
            />

            {/* Divider line */}
            <div
              className="absolute top-0 bottom-0 w-px bg-white/70 shadow-[0_0_6px_rgba(0,0,0,0.8)] pointer-events-none"
              style={{ left: `${splitPos}%` }}
            />
            {/* Handle knob */}
            <div
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-white/90 shadow-lg flex items-center justify-center pointer-events-none"
              style={{ left: `${splitPos}%` }}
            >
              <span className="text-slate-800 text-[10px] font-bold select-none">⇔</span>
            </div>

            {/* Labels */}
            <span className="absolute top-2 left-3 text-[10px] font-medium bg-black/60 text-white px-2 py-0.5 rounded pointer-events-none">
              Undistorted
            </span>
            <span className="absolute top-2 right-3 text-[10px] font-medium bg-black/60 text-white px-2 py-0.5 rounded pointer-events-none">
              Original
            </span>
          </>
        )}

        {!images && !loading && !error && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-xs">
            Select a frame to preview
          </div>
        )}
      </div>

      {/* Slider fallback */}
      <div className="space-y-1">
        <input
          type="range"
          min={0}
          max={100}
          value={splitPos}
          onChange={(e) => setSplitPos(Number(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="flex justify-between text-[10px] text-slate-500">
          <span>← Undistorted</span>
          <span>Original →</span>
        </div>
      </div>
    </div>
  );
}
