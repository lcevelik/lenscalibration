import { useEffect, useRef } from 'react';
import type { ScoredFrame } from '../types';

const GRID_COLS = 12;
const GRID_ROWS = 8;

interface Props {
  frames: ScoredFrame[];
  imageSize: [number, number];
}

function coverageColor(count: number): string {
  if (count === 0) return '#1e293b';
  const t = Math.min(count / 5, 1);
  // red → yellow → green
  if (t < 0.5) {
    const u = t * 2;
    const r = Math.round(239 + (251 - 239) * u);
    const g = Math.round(68  + (191 - 68)  * u);
    const b = Math.round(68  + (36  - 68)  * u);
    return `rgb(${r},${g},${b})`;
  } else {
    const u = (t - 0.5) * 2;
    const r = Math.round(251 + (16  - 251) * u);
    const g = Math.round(191 + (185 - 191) * u);
    const b = Math.round(36  + (129 - 36)  * u);
    return `rgb(${r},${g},${b})`;
  }
}

export default function CoverageMap({ frames, imageSize }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imgW, imgH] = imageSize;

  const foundFrames = frames.filter(f => f.found && f.corners.length > 0);
  const cellW = imgW / GRID_COLS;
  const cellH = imgH / GRID_ROWS;

  // Build grid: count[row][col] = number of frames with a corner in that cell
  const grid: number[][] = Array.from({ length: GRID_ROWS }, () => new Array(GRID_COLS).fill(0));
  for (const frame of foundFrames) {
    const seen = new Set<number>();
    for (const [x, y] of frame.corners) {
      const col = Math.min(GRID_COLS - 1, Math.floor(x / cellW));
      const row = Math.min(GRID_ROWS - 1, Math.floor(y / cellH));
      const key = row * GRID_COLS + col;
      if (!seen.has(key)) {
        seen.add(key);
        grid[row][col]++;
      }
    }
  }

  const coveredCells = grid.flat().filter(c => c > 0).length;
  const totalCells   = GRID_COLS * GRID_ROWS;
  const coveragePct  = Math.round((coveredCells / totalCells) * 100);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cw = canvas.width;
    const ch = canvas.height;
    const pw = cw / GRID_COLS;
    const ph = ch / GRID_ROWS;

    ctx.clearRect(0, 0, cw, ch);

    for (let row = 0; row < GRID_ROWS; row++) {
      for (let col = 0; col < GRID_COLS; col++) {
        ctx.fillStyle = coverageColor(grid[row][col]);
        ctx.fillRect(col * pw + 1, row * ph + 1, pw - 2, ph - 2);
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(15,23,42,0.6)';
    ctx.lineWidth = 1;
    for (let c = 1; c < GRID_COLS; c++) {
      ctx.beginPath(); ctx.moveTo(c * pw, 0); ctx.lineTo(c * pw, ch); ctx.stroke();
    }
    for (let r = 1; r < GRID_ROWS; r++) {
      ctx.beginPath(); ctx.moveTo(0, r * ph); ctx.lineTo(cw, r * ph); ctx.stroke();
    }
  }, [frames, imageSize]);

  return (
    <div className="rounded-xl bg-slate-800 border border-slate-700 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Corner Coverage
        </h3>
        <span className={`text-sm font-bold tabular-nums ${
          coveragePct >= 70 ? 'text-emerald-400' :
          coveragePct >= 40 ? 'text-yellow-400'  : 'text-red-400'
        }`}>
          {coveragePct}%
        </span>
      </div>

      <canvas
        ref={canvasRef}
        width={GRID_COLS * 32}
        height={GRID_ROWS * 32}
        className="w-full rounded-lg"
        style={{ imageRendering: 'pixelated' }}
      />

      {/* Legend */}
      <div className="flex items-center gap-3 text-[10px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: '#1e293b', border: '1px solid #334155' }} />
          No corners
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm inline-block bg-red-500" />
          1–2 frames
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm inline-block bg-yellow-500" />
          3–4 frames
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm inline-block bg-emerald-500" />
          5+ frames
        </span>
      </div>

      {foundFrames.length === 0 && (
        <p className="text-xs text-slate-600 text-center py-2">Score frames to see coverage</p>
      )}
    </div>
  );
}
