import { useEffect, useState } from 'react';
import type { CalibrationResult } from '../types';

interface Props {
  result: CalibrationResult;
  imageSize: [number, number];
  ws: WebSocket | null;
}

type ExportStatus = 'idle' | 'loading' | 'success' | 'error';

interface ExportFormat {
  key: 'ue5_ulens' | 'opencv_xml' | 'stmap_exr' | 'json';
  label: string;
  ext: string;
  description: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EXPORT_FORMATS: ExportFormat[] = [
  { key: 'ue5_ulens',  label: 'UE5 .ulens',   ext: 'ulens', description: 'Unreal Engine 5 CameraCalibration plugin' },
  { key: 'opencv_xml', label: 'OpenCV XML',    ext: 'xml',   description: 'Ventuz / FreeD compatible FileStorage XML' },
  { key: 'stmap_exr',  label: 'STmap EXR',     ext: 'exr',   description: '32-bit float UV remap for compositing' },
  { key: 'json',       label: 'JSON',          ext: 'json',  description: 'Full calibration data with metadata' },
];

// ---------------------------------------------------------------------------
// Camera sensor presets
// ---------------------------------------------------------------------------

interface SensorPreset {
  label:  string;
  w:      number;
  h:      number;
  group?: string;
}

const SENSOR_PRESETS: SensorPreset[] = [
  // Sony Venice
  { group: 'Sony Venice',         label: 'Venice — FF 6K (36.0 × 24.0)',                w: 36.0,  h: 24.0  },
  // Sony Venice 2
  { group: 'Sony Venice 2',       label: 'Venice 2 — FF 8K (35.9 × 24.0)',              w: 35.9,  h: 24.0  },
  { group: 'Sony Venice 2',       label: 'Venice 2 — FF 6K (36.0 × 24.0)',              w: 36.0,  h: 24.0  },
  { group: 'Sony Venice 2',       label: 'Venice 2 — S35 6K (26.2 × 14.7)',             w: 26.2,  h: 14.7  },
  { group: 'Sony Venice 2',       label: 'Venice 2 — S16 (14.6 × 8.2)',                 w: 14.6,  h: 8.2   },
  // Sony Burano
  { group: 'Sony Burano',         label: 'Burano — FF 8K (35.9 × 24.0)',               w: 35.9,  h: 24.0  },
  { group: 'Sony Burano',         label: 'Burano — S35 (26.2 × 14.7)',                  w: 26.2,  h: 14.7  },
  { group: 'Sony Burano',         label: 'Burano — S16 (14.6 × 8.2)',                   w: 14.6,  h: 8.2   },
  // Sony Broadcast B4
  { group: 'Sony Broadcast B4',   label: 'HDC-series 2/3" HD B4 (9.59 × 5.39)',        w: 9.59,  h: 5.39  },
  { group: 'Sony Broadcast B4',   label: 'HDC-F5500 / HDC-3500 4K 2/3" (9.59 × 5.39)', w: 9.59,  h: 5.39  },
  { group: 'Sony Broadcast B4',   label: 'HDC-5500 / HDW-series 2/3" (9.59 × 5.39)',   w: 9.59,  h: 5.39  },
];

const DC_NAMES_5  = ['k1', 'k2', 'p1', 'p2', 'k3'];
const DC_NAMES_8  = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'];
const DC_NAMES_14 = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4', 'τx', 'τy'];

const DC_TIPS: Record<string, string> = {
  k1: 'Radial distortion (1st order). Negative = barrel distortion (lines bow outward). Positive = pincushion (lines bow inward).',
  k2: 'Radial distortion (2nd order). Corrects more complex radial warping. Usually much smaller than k1.',
  k3: 'Radial distortion (3rd order). Only meaningful for extreme wide-angle lenses. Often kept at 0.',
  k4: 'Rational model denominator radial term (1st order). Part of the extended rational distortion model.',
  k5: 'Rational model denominator radial term (2nd order).',
  k6: 'Rational model denominator radial term (3rd order).',
  p1: 'Tangential distortion — compensates for the lens not being perfectly parallel to the sensor.',
  p2: 'Tangential distortion (perpendicular axis). Should be close to p1 in magnitude.',
  s1: 'Thin prism distortion coefficient. Used for sensors with prism-like elements.',
  s2: 'Thin prism distortion coefficient.',
  s3: 'Thin prism distortion coefficient.',
  s4: 'Thin prism distortion coefficient.',
  'τx': 'Tilt model X component for sensor tilt compensation.',
  'τy': 'Tilt model Y component for sensor tilt compensation.',
};

const CAM_TIPS: Record<string, string> = {
  fx: 'Horizontal focal length in pixels. Larger = more telephoto. Converts 3D world distances to pixels along X.',
  fy: 'Vertical focal length in pixels. Should be very close to fx; a large difference suggests lens shear.',
  cx: 'Principal point X — where the optical axis intersects the image plane. Ideally the image center width/2.',
  cy: 'Principal point Y — where the optical axis intersects the image plane. Ideally the image center height/2.',
};

const CONFIDENCE_STYLE: Record<string, string> = {
  excellent: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  good:      'bg-blue-500/15    text-blue-400    border-blue-500/30',
  marginal:  'bg-yellow-500/15  text-yellow-400  border-yellow-500/30',
  poor:      'bg-red-500/15     text-red-400     border-red-500/30',
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function InfoTip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span className="ml-1 cursor-help text-slate-600 hover:text-slate-400 text-[11px]">(?)</span>
      {show && (
        <div className="absolute z-30 bottom-full left-1/2 -translate-x-1/2 mb-2 w-60 bg-slate-700 border border-slate-600 text-slate-200 text-xs rounded-lg p-3 shadow-xl pointer-events-none leading-relaxed">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-700" />
        </div>
      )}
    </span>
  );
}

function filename(p: string) {
  return p.replace(/\\/g, '/').split('/').pop() ?? p;
}

// ---------------------------------------------------------------------------
// Distortion curve SVG chart
// ---------------------------------------------------------------------------

function DistortionCurve({ k1, k2, k3 }: { k1: number; k2: number; k3: number }) {
  const W = 320, H = 180, PAD = 32;
  const n = 100;
  const rMax = 1.5; // normalised radius (1.0 = corner)
  const dr = rMax / n;

  // Compute r'/r for each r
  const points: [number, number][] = [];
  let yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i <= n; i++) {
    const r = i * dr;
    const r2 = r * r;
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    const rPrime = r === 0 ? 0 : r * (1 + k1 * r2 + k2 * r4 + k3 * r6);
    const ratio = r === 0 ? 1 : rPrime / r;
    points.push([r, ratio]);
    if (ratio < yMin) yMin = ratio;
    if (ratio > yMax) yMax = ratio;
  }

  // Add margin to y range
  const yPad = (yMax - yMin) * 0.15 || 0.05;
  yMin -= yPad;
  yMax += yPad;

  const xScale = (v: number) => PAD + (v / rMax) * (W - 2 * PAD);
  const yScale = (v: number) => H - PAD - ((v - yMin) / (yMax - yMin)) * (H - 2 * PAD);

  const pathD = points.map(([r, y], i) =>
    `${i === 0 ? 'M' : 'L'} ${xScale(r).toFixed(1)} ${yScale(y).toFixed(1)}`
  ).join(' ');

  // Undistorted reference line (r'/r = 1)
  const refY = yScale(1);

  // Barrel/pincushion label
  const midRatio = points[Math.floor(n * 0.5)]?.[1] ?? 1;
  const distType = midRatio < 0.998 ? 'barrel' : midRatio > 1.002 ? 'pincushion' : 'minimal';

  return (
    <div className="bg-slate-900 rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
          Radial Distortion Profile
        </span>
        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
          distType === 'barrel' ? 'bg-blue-500/15 text-blue-400' :
          distType === 'pincushion' ? 'bg-yellow-500/15 text-yellow-400' :
          'bg-slate-700 text-slate-400'
        }`}>
          {distType}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 180 }}>
        {/* Grid lines */}
        {[0, 0.5, 1.0, 1.5].map(v => v <= rMax && (
          <line key={v} x1={xScale(v)} y1={PAD} x2={xScale(v)} y2={H - PAD}
            stroke="#334155" strokeWidth={0.5} />
        ))}
        {/* Undistorted reference */}
        <line x1={PAD} y1={refY} x2={W - PAD} y2={refY}
          stroke="#475569" strokeWidth={1} strokeDasharray="4 3" />
        {/* Distortion curve */}
        <path d={pathD} fill="none" stroke="#60a5fa" strokeWidth={2} />
        {/* Axes */}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#64748b" strokeWidth={1} />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#64748b" strokeWidth={1} />
        {/* X labels */}
        <text x={xScale(0)} y={H - 8} textAnchor="middle" fill="#64748b" fontSize={9}>0</text>
        <text x={xScale(1)} y={H - 8} textAnchor="middle" fill="#64748b" fontSize={9}>1.0</text>
        {/* Y labels */}
        <text x={PAD - 4} y={yScale(1) + 3} textAnchor="end" fill="#64748b" fontSize={9}>1.0</text>
        <text x={PAD - 4} y={yScale(yMin) + 3} textAnchor="end" fill="#64748b" fontSize={9}>{yMin.toFixed(3)}</text>
        <text x={PAD - 4} y={yScale(yMax) + 3} textAnchor="end" fill="#64748b" fontSize={9}>{yMax.toFixed(3)}</text>
        {/* Axis titles */}
        <text x={W / 2} y={H - 1} textAnchor="middle" fill="#64748b" fontSize={9}>r (normalised radius)</text>
        <text x={6} y={H / 2} textAnchor="middle" fill="#64748b" fontSize={9}
          transform={`rotate(-90, 6, ${H / 2})`}>r′/r</text>
      </svg>
      <p className="text-[10px] text-slate-500 mt-1">
        r′/r = 1 means no distortion. Below 1 = barrel; above 1 = pincushion.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ResultPanel({ result, imageSize, ws }: Props) {
  const [exportStates, setExportStates] = useState<Record<string, ExportStatus>>(
    Object.fromEntries(EXPORT_FORMATS.map((f) => [f.key, 'idle']))
  );
  const [exportPaths, setExportPaths] = useState<Record<string, string>>({});

  // UE5 .ulens extra metadata
  const [lensName, setLensName]               = useState('Lens');
  const [sensorWidthMm, setSensorWidthMm]     = useState('');
  const [sensorHeightMm, setSensorHeightMm]   = useState('');

  // Listen for export_result messages
  useEffect(() => {
    if (!ws) return;
    const handler = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.action !== 'export_result') return;
        const fmt: string = msg.format;
        setExportStates((prev) => ({ ...prev, [fmt]: msg.success ? 'success' : 'error' }));
        if (msg.success) {
          setExportPaths((prev) => ({ ...prev, [fmt]: msg.output_path }));
        }
      } catch {
        // ignore
      }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  const sendExport = async (fmt: ExportFormat) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Ask user where to save via the native OS dialog (Electron only)
    let outputPath = `calibration.${fmt.ext}`;
    if (window.electronAPI?.showSaveDialog) {
      const dlg = await window.electronAPI.showSaveDialog({
        defaultPath: outputPath,
        filters: [
          { name: fmt.label, extensions: [fmt.ext] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (dlg.canceled || !dlg.filePath) return;
      outputPath = dlg.filePath;
    }

    setExportStates((prev) => ({ ...prev, [fmt.key]: 'loading' }));
    const msg: Record<string, unknown> = {
      action: 'export',
      format: fmt.key,
      output_path: outputPath,
      camera_matrix: result.camera_matrix,
      dist_coeffs: result.dist_coeffs,
      fov_x: result.fov_x,
      fov_y: result.fov_y,
      rms: result.rms,
      image_size: imageSize,
      metadata: {},
      squeeze_ratio: result.squeeze_ratio ?? 1,
      lens_type: result.lens_type ?? 'spherical',
    };
    if (fmt.key === 'ue5_ulens') {
      msg.lens_name        = lensName.trim() || 'Lens';
      msg.sensor_width_mm  = parseFloat(sensorWidthMm)  || 0;
      msg.sensor_height_mm = parseFloat(sensorHeightMm) || 0;
    }
    ws.send(JSON.stringify(msg));
  };

  // RMS colour
  const rmsColor =
    result.rms < 0.3 ? 'text-emerald-400' :
    result.rms < 1.0 ? 'text-yellow-400'  : 'text-red-400';

  // Camera matrix
  const cm = result.camera_matrix;
  const fx = cm[0][0], fy = cm[1][1], cx = cm[0][2], cy = cm[1][2];

  // Distortion labels
  const dc = result.dist_coeffs;
  const dcNames =
    dc.length >= 14 ? DC_NAMES_14 :
    dc.length >= 8  ? DC_NAMES_8  : DC_NAMES_5;

  // Per-image error chart
  const errors = result.per_image_errors;
  const maxErr = Math.max(...errors.map((e) => e.error), 0.001);
  const meanErr = errors.reduce((s, e) => s + e.error, 0) / (errors.length || 1);

  const barVariant = (entry: { error: number; outlier: boolean }) =>
    entry.outlier         ? 'bar-bad' :
    entry.error > meanErr ? 'bar-warn' : 'bar-good';

  return (
    <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-6">
      <h2 className="text-sm font-semibold text-slate-300 tracking-wide uppercase">
        Calibration Results
      </h2>

      {/* RMS + Confidence row */}
      <div className="flex items-end gap-6">
        <div>
          <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">
            RMS reprojection error
            <InfoTip text="Average distance (in pixels) between detected corners and their re-projected positions. < 0.3 is excellent for virtual production; < 1.0 is generally usable." />
          </p>
          <span className={`text-5xl font-bold tabular-nums ${rmsColor}`}>
            {result.rms.toFixed(3)}
          </span>
          <span className="text-slate-500 ml-1 text-sm">px</span>
        </div>

        <div className="space-y-1.5">
          <span
            className={`inline-block px-3 py-1 rounded-full text-sm font-semibold border capitalize ${CONFIDENCE_STYLE[result.confidence]}`}
          >
            {result.confidence}
          </span>
          <p className="text-xs text-slate-500">
            {result.used_frames} frames used
            {result.skipped_frames > 0 && `, ${result.skipped_frames} skipped`}
          </p>
          <p className="text-xs text-slate-500">
            FOV {result.fov_x.toFixed(1)}° × {result.fov_y.toFixed(1)}°
          </p>
        </div>
      </div>

      {/* Camera matrix */}
      <div>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Camera Matrix
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {(
            [
              ['fx', fx],
              ['fy', fy],
              ['cx', cx],
              ['cy', cy],
            ] as [string, number][]
          ).map(([name, val]) => (
            <div key={name} className="bg-slate-900 rounded-lg px-3 py-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 font-mono">
                  {name}
                  <InfoTip text={CAM_TIPS[name]} />
                </span>
                <span className="text-sm font-mono text-slate-200 tabular-nums">
                  {val.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Distortion coefficients */}
      <div>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Distortion Coefficients
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {dc.slice(0, dcNames.length).map((val, i) => {
            const name = dcNames[i];
            return (
              <div key={name} className="bg-slate-900 rounded-lg px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-mono">
                    {name}
                    {DC_TIPS[name] && <InfoTip text={DC_TIPS[name]} />}
                  </span>
                  <span
                    className={`text-xs font-mono tabular-nums ${
                      Math.abs(val) > 0.5 ? 'text-yellow-400' : 'text-slate-200'
                    }`}
                  >
                    {val.toFixed(5)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Distortion curve */}
      {dc.length >= 3 && (
        <DistortionCurve
          k1={dc[0] ?? 0}
          k2={dc[1] ?? 0}
          k3={dc[4] ?? 0}
        />
      )}

      {/* Per-image error chart */}
      {errors.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Per-image Reprojection Error
            <InfoTip text="Each bar shows the RMS reprojection error for that frame. Red bars are statistical outliers (> 1.5× mean). Consider excluding outlier frames and re-running calibration." />
          </h3>
          <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
            {errors.map((entry, i) => (
              <div key={i} className="flex items-center gap-2 text-[11px]">
                <span
                  className="w-28 shrink-0 truncate text-slate-500 text-right font-mono"
                  title={entry.path}
                >
                  {filename(entry.path)}
                </span>
                <progress
                    className={`result-error-bar ${barVariant(entry)}`}
                    value={entry.error}
                    max={maxErr}
                  />
                <span
                  className={`w-14 shrink-0 text-right font-mono tabular-nums ${
                    entry.outlier ? 'text-red-400' : 'text-slate-400'
                  }`}
                >
                  {entry.error.toFixed(4)}
                </span>
                {entry.outlier && (
                  <span className="text-red-500 text-[10px]" title="Outlier">⚠</span>
                )}
              </div>
            ))}
          </div>
          {/* Legend */}
          <div className="flex gap-4 mt-2">
            {[
              { color: 'bg-emerald-500', label: '≤ mean' },
              { color: 'bg-yellow-500',  label: '> mean' },
              { color: 'bg-red-500',     label: 'outlier' },
            ].map(({ color, label }) => (
              <div key={label} className="flex items-center gap-1.5 text-[10px] text-slate-500">
                <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
                {label}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export buttons */}
      <div>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Export
        </h3>

        {/* UE5 .ulens metadata */}
        <div className="mb-3 p-3 rounded-lg bg-slate-900 border border-slate-700 space-y-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">UE5 .ulens metadata</p>

          {/* Camera sensor preset picker */}
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] text-slate-500">Camera preset</span>
            <select
              aria-label="Camera sensor preset"
              defaultValue=""
              onChange={e => {
                const preset = SENSOR_PRESETS.find(p => p.label === e.target.value);
                if (preset) {
                  setSensorWidthMm(String(preset.w));
                  setSensorHeightMm(String(preset.h));
                }
              }}
              className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
            >
              <option value="" disabled>Select camera to fill sensor size…</option>
              {(['Sony Venice', 'Sony Venice 2', 'Sony Burano'] as const).map(group => (
                <optgroup key={group} label={group}>
                  {SENSOR_PRESETS.filter(p => p.group === group).map(p => (
                    <option key={p.label} value={p.label}>{p.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          <div className="flex gap-2 flex-wrap">
            <label className="flex flex-col gap-0.5 flex-1 min-w-[120px]">
              <span className="text-[10px] text-slate-500">Lens name</span>
              <input
                value={lensName}
                onChange={e => setLensName(e.target.value)}
                placeholder="e.g. Premista28-100mm"
                className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
              />
            </label>
            <label className="flex flex-col gap-0.5 w-20">
              <span className="text-[10px] text-slate-500">Sensor W mm</span>
              <input
                type="number"
                value={sensorWidthMm}
                onChange={e => setSensorWidthMm(e.target.value)}
                placeholder="36.0"
                className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
              />
            </label>
            <label className="flex flex-col gap-0.5 w-20">
              <span className="text-[10px] text-slate-500">Sensor H mm</span>
              <input
                type="number"
                value={sensorHeightMm}
                onChange={e => setSensorHeightMm(e.target.value)}
                placeholder="24.0"
                className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
              />
            </label>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {EXPORT_FORMATS.map((fmt) => {
            const state = exportStates[fmt.key];
            const outPath = exportPaths[fmt.key];
            return (
              <button
                key={fmt.key}
                onClick={() => sendExport(fmt)}
                disabled={state === 'loading' || !ws}
                title={fmt.description}
                className={`
                  flex items-center justify-between gap-2 px-3 py-2.5 rounded-lg border text-xs text-left
                  transition-colors focus:outline-none
                  ${state === 'success'
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                    : state === 'error'
                    ? 'bg-red-500/10 border-red-500/30 text-red-400'
                    : state === 'loading'
                    ? 'bg-slate-700 border-slate-600 text-slate-400 cursor-wait'
                    : 'bg-slate-700 border-slate-600 text-slate-200 hover:bg-slate-600 hover:border-slate-500 cursor-pointer'
                  }
                `}
              >
                <span className="font-medium">{fmt.label}</span>
                <span className="shrink-0 text-base leading-none">
                  {state === 'loading' && (
                    <span className="inline-block w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  )}
                  {state === 'success' && '✓'}
                  {state === 'error'   && '✗'}
                </span>
              </button>
            );
          })}
        </div>

        {/* Show saved paths */}
        {Object.entries(exportPaths).map(([fmt, p]) => (
          <p key={fmt} className="mt-1 text-[10px] text-slate-500 font-mono truncate" title={p}>
            {fmt}: {p}
          </p>
        ))}
      </div>
    </div>
  );
}
