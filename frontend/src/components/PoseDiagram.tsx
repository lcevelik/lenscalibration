/**
 * PoseDiagram — SVG visual guide showing where/how to hold the calibration
 * board in the camera frame for each required pose.
 *
 * Coordinate space: 100 × 70 viewBox (proportional to 16:9 frame).
 * Each pose definition contains:
 *   pts   — four [x,y] corners of the board polygon (perspective-distorted
 *           to convey tilt direction)
 *   arrow — optional arc-arrow drawn on top to reinforce rotation axis
 */

import React from 'react';

// ---------------------------------------------------------------------------
// Pose shapes
// ---------------------------------------------------------------------------

type Pt = [number, number];

interface PoseShape {
  pts:   Pt[];            // board quad corners, clockwise from top-left
  arrow?: ArrowDef;       // optional rotation hint
  label?: string;         // short axis label shown inside frame
}

interface ArrowDef {
  // SVG arc path from → to, drawn as a curved arrow
  d:     string;
  tip:   Pt;              // arrowhead tip point
  angle: number;          // rotation of arrowhead (degrees)
}

const SHAPES: Record<string, PoseShape> = {
  center_flat: {
    pts: [[28,21],[72,21],[72,49],[28,49]],
  },
  center_tilted_h: {
    // board rotated ~30° around vertical axis — right side foreshortened
    pts: [[28,19],[68,24],[68,46],[28,51]],
    arrow: {
      d:     'M 22,35 A 14,14 0 0,1 78,35',
      tip:   [78, 35],
      angle: 80,
    },
  },
  center_tilted_v: {
    // top tilted away — top edge shorter / higher
    pts: [[32,23],[68,23],[72,49],[28,49]],
    arrow: {
      d:     'M 50,14 A 14,14 0 0,1 50,56',
      tip:   [50, 56],
      angle: 170,
    },
  },
  corner_tl: {
    pts: [[2,2],[38,2],[38,27],[2,27]],
  },
  corner_tr: {
    pts: [[62,2],[98,2],[98,27],[62,27]],
  },
  corner_bl: {
    pts: [[2,43],[38,43],[38,68],[2,68]],
  },
  corner_br: {
    pts: [[62,43],[98,43],[98,68],[62,68]],
  },
  close_up: {
    // board fills almost the entire frame
    pts: [[5,4],[95,4],[95,66],[5,66]],
  },
  strong_tilt: {
    // heavy diagonal tilt — board shown almost edge-on at an angle
    pts: [[20,15],[55,9],[76,55],[41,61]],
    arrow: {
      d:     'M 28,62 A 22,22 0 0,1 72,8',
      tip:   [72, 8],
      angle: 45,
    },
  },
  top_or_bottom: {
    // board along top edge (arrow hints it can also go bottom)
    pts: [[14,2],[86,2],[86,24],[14,24]],
    arrow: {
      d:     'M 50,30 L 50,42',
      tip:   [50, 42],
      angle: 180,
    },
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function polyPoints(pts: Pt[]): string {
  return pts.map(([x, y]) => `${x},${y}`).join(' ');
}

function centroid(pts: Pt[]): Pt {
  const x = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const y = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return [x, y];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  poseId:     string;
  size?:      'sm' | 'lg';
  satisfied?: boolean;
}

export default function PoseDiagram({ poseId, size = 'sm', satisfied = false }: Props) {
  const shape = SHAPES[poseId];
  if (!shape) return null;

  const uid    = `pd-${poseId}`;
  const isLg   = size === 'lg';
  const px     = isLg ? 180 : 90;
  const py     = isLg ? 126 : 63;

  const boardColor  = satisfied ? '#10b981' : '#818cf8'; // emerald : indigo
  const boardFill   = satisfied ? 'rgba(16,185,129,0.18)' : 'rgba(129,140,248,0.18)';
  const arrowColor  = satisfied ? '#6ee7b7' : '#a5b4fc';
  const gridColor   = '#1e293b';
  const frameStroke = '#334155';

  const [cx, cy] = centroid(shape.pts);

  return (
    <svg
      viewBox="0 0 100 70"
      width={px}
      height={py}
      className="rounded overflow-hidden shrink-0 bg-slate-950"
      aria-label={`Pose diagram for ${poseId}`}
    >
      {/* ── Defs ─────────────────────────────────────────── */}
      <defs>
        {/* Checkerboard fill pattern for the board */}
        <pattern id={`${uid}-chk`} x={shape.pts[0][0]} y={shape.pts[0][1]}
          width="12" height="12" patternUnits="userSpaceOnUse"
        >
          <rect width="6"  height="6"  fill="rgba(255,255,255,0.22)" />
          <rect x="6" width="6" height="6" fill="rgba(0,0,0,0.10)" />
          <rect y="6" width="6" height="6" fill="rgba(0,0,0,0.10)" />
          <rect x="6" y="6" width="6" height="6" fill="rgba(255,255,255,0.22)" />
        </pattern>

        {/* Clip path matching the board polygon */}
        <clipPath id={`${uid}-clip`}>
          <polygon points={polyPoints(shape.pts)} />
        </clipPath>

        {/* Arrowhead marker */}
        <marker id={`${uid}-arr`} markerWidth="6" markerHeight="6"
          refX="3" refY="3" orient="auto"
        >
          <path d="M0,0 L6,3 L0,6 Z" fill={arrowColor} />
        </marker>
      </defs>

      {/* ── Frame background ─────────────────────────────── */}
      <rect width="100" height="70" className="fill-slate-950" fill="#0f172a" />

      {/* 3×3 region grid */}
      <line x1="33.3" y1="0" x2="33.3" y2="70" stroke={gridColor} strokeWidth="0.6" />
      <line x1="66.7" y1="0" x2="66.7" y2="70" stroke={gridColor} strokeWidth="0.6" />
      <line x1="0"  y1="23.3" x2="100" y2="23.3" stroke={gridColor} strokeWidth="0.6" />
      <line x1="0"  y1="46.7" x2="100" y2="46.7" stroke={gridColor} strokeWidth="0.6" />

      {/* Frame border */}
      <rect x="0.5" y="0.5" width="99" height="69" fill="none"
        stroke={frameStroke} strokeWidth="1" />

      {/* ── Board shape ──────────────────────────────────── */}

      {/* Checkerboard fill clipped to board polygon */}
      <rect width="100" height="70"
        fill={`url(#${uid}-chk)`}
        clipPath={`url(#${uid}-clip)`}
      />

      {/* Tinted overlay for color tint */}
      <polygon
        points={polyPoints(shape.pts)}
        fill={boardFill}
        stroke={boardColor}
        strokeWidth={isLg ? 1.5 : 1}
      />

      {/* Board centre dot */}
      <circle cx={cx} cy={cy} r={isLg ? 2 : 1.2} fill={boardColor} opacity={0.9} />

      {/* ── Tilt/rotation arrow ──────────────────────────── */}
      {shape.arrow && (
        <path
          d={shape.arrow.d}
          fill="none"
          stroke={arrowColor}
          strokeWidth={isLg ? 1.4 : 0.9}
          strokeDasharray={isLg ? '3 2' : '2 1.5'}
          markerEnd={`url(#${uid}-arr)`}
          opacity={0.85}
        />
      )}

      {/* ── Satisfied tick ───────────────────────────────── */}
      {satisfied && (
        <text x="97" y="68" textAnchor="end" dominantBaseline="auto"
          fontSize={isLg ? 9 : 6} fill="#10b981" fontWeight="bold"
        >
          ✓
        </text>
      )}
    </svg>
  );
}
