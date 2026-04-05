import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, CameraSettings, CalibrationResult, ConnStatus, LensSettings, ScoredFrame } from './types';
import type { ZoomCalibResult, ZoomFlResult } from './components/GuidedCapture';
import CoverageMap from './components/CoverageMap';
import DropZone from './components/DropZone';
import FileCalibration from './components/FileCalibration';
import FrameGrid from './components/FrameGrid';
import GuidedCapture from './components/GuidedCapture';
import ResultPanel from './components/ResultPanel';
import BoardPresetSelector from './components/BoardPresetSelector';
import UndistortPreview from './components/UndistortPreview';

interface DeviceBrand   { id: string; name: string; icon: string; is_capture_card: boolean; }
interface CaptureDevice { index: number; name: string; brand: DeviceBrand; }

const BRAND_COLOR: Record<string, string> = {
  blackmagic: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  aja:        'bg-red-500/20    text-red-300    border-red-500/30',
  bluefish444:'bg-blue-500/20   text-blue-300   border-blue-500/30',
  magewell:   'bg-purple-500/20 text-purple-300 border-purple-500/30',
  datapath:   'bg-orange-500/20 text-orange-300 border-orange-500/30',
  deltacast:  'bg-cyan-500/20   text-cyan-300   border-cyan-500/30',
  generic:    'bg-slate-700     text-slate-400  border-slate-600',
};

// Minimal frame shape that UndistortPreview needs
type PreviewFrame = { path: string; quality: 'good' | 'warn' | 'fail'; sharpness: number; coverage: number; index: number };

const SENSOR_PRESETS = [
  // CineAlta
  { label: 'Venice — FF 6K (36.0 × 24.0)',                 w: 36.0,  h: 24.0  },
  { label: 'Venice 2 / Burano — FF 8K (35.9 × 24.0)',      w: 35.9,  h: 24.0  },
  { label: 'Venice 2 / Burano — S35 (26.2 × 14.7)',        w: 26.2,  h: 14.7  },
  { label: 'Venice 2 / Burano — S16 (14.6 × 8.2)',         w: 14.6,  h: 8.2   },
  // Broadcast B4
  { label: 'HDC-series 2/3" HD B4 (9.59 × 5.39)',          w: 9.59,  h: 5.39  },
  { label: 'HDC-F5500 / HDC-3500 4K 2/3" (9.59 × 5.39)',   w: 9.59,  h: 5.39  },
  { label: 'HDC-5500 / HDW-series 2/3" (9.59 × 5.39)',     w: 9.59,  h: 5.39  },
];

const STATUS_DOT: Record<ConnStatus, string> = {
  connecting:   'bg-yellow-400 animate-pulse',
  connected:    'bg-emerald-400',
  disconnected: 'bg-slate-500',
  error:        'bg-red-400',
};

const ZOOM_CONFIDENCE_COLOR: Record<string, string> = {
  excellent: 'text-emerald-400',
  good:      'text-blue-400',
  marginal:  'text-yellow-400',
  poor:      'text-red-400',
};

// ---------------------------------------------------------------------------
// Zoom Results Panel (used in the Results tab after multi-FL calibration)
// ---------------------------------------------------------------------------

/** Convert a ZoomFlResult to a CalibrationResult so ResultPanel can render it. */
function flResultToCalibResult(r: ZoomFlResult, imageSize: [number, number]): CalibrationResult {
  const fx = r.fx_px || 1, fy = r.fy_px || 1;
  const fov_x = 2 * Math.atan(imageSize[0] / (2 * fx)) * (180 / Math.PI);
  const fov_y = 2 * Math.atan(imageSize[1] / (2 * fy)) * (180 / Math.PI);
  return {
    rms: r.rms ?? 0,
    camera_matrix: r.camera_matrix,
    dist_coeffs: r.dist_coeffs,
    fov_x,
    fov_y,
    confidence: (r.confidence ?? 'poor') as CalibrationResult['confidence'],
    per_image_errors: r.per_image_errors ?? [],
    used_frames: r.used_frames ?? 0,
    skipped_frames: 0,
  };
}

interface ZoomResultsPanelProps {
  result: ZoomCalibResult;
  imageSize: [number, number];
  ws: WebSocket | null;
  lensSettings: LensSettings;
  cameraSettings: CameraSettings;
  exportStatus: 'idle' | 'loading' | 'success' | 'error';
  exportPath: string | null;
  onExportStatusChange: (s: 'idle' | 'loading' | 'success' | 'error') => void;
  selectedFlMm: number | null;
  onSelectFl: (fl: number | null) => void;
}

function ZoomResultsPanel({ result, imageSize, ws, lensSettings, cameraSettings, exportStatus, exportPath, onExportStatusChange, selectedFlMm, onSelectFl }: ZoomResultsPanelProps) {
  const [showNodalTable, setShowNodalTable] = useState(false);

  const doExport = async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    let outputPath = 'calibration_zoom.ulens';
    if (window.electronAPI?.showSaveDialog) {
      const dlg = await window.electronAPI.showSaveDialog({
        defaultPath: outputPath,
        filters: [{ name: 'UE5 Lens File', extensions: ['ulens'] }, { name: 'All Files', extensions: ['*'] }],
      });
      if (dlg.canceled || !dlg.filePath) return;
      outputPath = dlg.filePath;
    }
    onExportStatusChange('loading');
    ws.send(JSON.stringify({
      action: 'export', format: 'ue5_ulens_zoom', output_path: outputPath,
      fl_results: result.fl_results, nodal_offsets_mm: result.nodal_offsets_mm,
      image_size: imageSize, lens_name: cameraSettings.lensName.trim() || 'Lens',
      sensor_width_mm: parseFloat(cameraSettings.sensorWidthMm) || 0,
      sensor_height_mm: parseFloat(cameraSettings.sensorHeightMm) || 0,
      nodal_preset: cameraSettings.nodalPreset.trim(),
      lens_type: lensSettings.lensType,
      squeeze_ratio: lensSettings.squeezeRatio,
    }));
  };

  const fls = result.fl_results.length;
  const avgRms = result.fl_results.filter(r => r.rms != null).reduce((s, r) => s + (r.rms ?? 0), 0) /
                 Math.max(1, result.fl_results.filter(r => r.rms != null).length);

  const selectedFlResult = selectedFlMm != null
    ? result.fl_results.find(r => r.focal_length_mm === selectedFlMm) ?? null
    : null;

  const interpRows = result.fl_interpolated ?? [];

  return (
    <div className="space-y-5">
      {result.error && (
        <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 text-sm text-red-400">{result.error}</div>
      )}

      {/* Summary badge row */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="rounded-full bg-blue-500/15 border border-blue-500/30 text-blue-400 text-xs font-semibold px-3 py-1">
          {fls} focal length{fls !== 1 ? 's' : ''}
        </span>
        <span className="rounded-full bg-slate-700 border border-slate-600 text-slate-300 text-xs font-semibold px-3 py-1">
          Avg RMS {avgRms.toFixed(3)} px
        </span>
        <span className="rounded-full bg-slate-700 border border-slate-600 text-slate-300 text-xs font-semibold px-3 py-1">
          {imageSize[0]}×{imageSize[1]}
        </span>
      </div>

      {/* Two-column layout when a FL is selected */}
      <div className={selectedFlResult ? 'grid grid-cols-2 gap-5 items-start' : ''}>

        {/* Left: Per-FL summary table */}
        <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-4">
          <h2 className="text-sm font-semibold text-slate-300 tracking-wide uppercase">Per Focal Length — click to inspect</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-700">
                  <th className="text-left pb-2 pr-4">FL (mm)</th>
                  <th className="text-right pb-2 pr-4">RMS</th>
                  <th className="text-right pb-2 pr-4">f (px)</th>
                  <th className="text-right pb-2 pr-4">k1</th>
                  <th className="text-right pb-2 pr-4">Nodal Δ</th>
                  <th className="text-right pb-2">Quality</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {result.fl_results.map(r => {
                  const nzKey = Number.isInteger(r.focal_length_mm) ? String(r.focal_length_mm) : r.focal_length_mm.toFixed(1);
                  const nz = result.nodal_offsets_mm[nzKey] ?? result.nodal_offsets_mm[String(r.focal_length_mm)];
                  const isSelected = selectedFlMm === r.focal_length_mm;
                  return (
                    <tr key={r.focal_length_mm}
                      onClick={() => !r.error && onSelectFl(isSelected ? null : r.focal_length_mm)}
                      className={`cursor-pointer transition-colors ${r.error ? 'opacity-50' : isSelected ? 'bg-blue-500/15' : 'hover:bg-slate-700/60'}`}>
                      <td className={`py-2 pr-4 font-semibold ${isSelected ? 'text-blue-300' : 'text-slate-200'}`}>{r.focal_length_mm}</td>
                      <td className={`py-2 pr-4 text-right font-mono tabular-nums ${r.rms == null ? 'text-red-400' : r.rms < 0.5 ? 'text-emerald-400' : r.rms < 1.0 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {r.rms != null ? r.rms.toFixed(3) : '—'}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-300">{r.fx_px ? r.fx_px.toFixed(0) : '—'}</td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-300">{r.dist_coeffs?.[0] != null ? r.dist_coeffs[0].toFixed(4) : '—'}</td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-400">{nz != null ? `${nz >= 0 ? '+' : ''}${nz.toFixed(1)} mm` : '—'}</td>
                      <td className={`py-2 text-right capitalize ${ZOOM_CONFIDENCE_COLOR[r.confidence] ?? 'text-slate-500'}`}>
                        {r.error ? 'error' : (r.confidence ?? '—')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Nodal Δ = Z-axis shift vs reference FL. Click a row to see full distortion details.
          </p>

          {/* Interpolated nodal offset table toggle */}
          {interpRows.length > 0 && (
            <div className="pt-2 border-t border-slate-700">
              <button type="button" onClick={() => setShowNodalTable(v => !v)}
                className="text-xs text-blue-400 hover:text-blue-300 font-semibold transition-colors">
                {showNodalTable ? '▾ Hide' : '▸ Show'} interpolated nodal offsets ({interpRows.length} FLs)
              </button>
              {showNodalTable && (
                <div className="mt-3 overflow-x-auto max-h-64 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-slate-800">
                      <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-700">
                        <th className="text-left pb-1.5 pr-4">FL (mm)</th>
                        <th className="text-right pb-1.5 pr-4">f (px)</th>
                        <th className="text-right pb-1.5 pr-4">k1</th>
                        <th className="text-right pb-1.5">Nodal Δ (mm)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/30">
                      {interpRows.map(r => (
                        <tr key={r.focal_length_mm} className="text-slate-400">
                          <td className="py-1 pr-4 font-mono">{r.focal_length_mm.toFixed(1)}</td>
                          <td className="py-1 pr-4 text-right font-mono">{r.fx_px.toFixed(0)}</td>
                          <td className="py-1 pr-4 text-right font-mono">{r.dist_coeffs?.[0]?.toFixed(4) ?? '—'}</td>
                          <td className={`py-1 text-right font-mono tabular-nums ${r.nodal_offset_z_mm >= 0 ? 'text-slate-300' : 'text-slate-300'}`}>
                            {r.nodal_offset_z_mm >= 0 ? '+' : ''}{r.nodal_offset_z_mm.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* Export */}
          <div className="pt-3 border-t border-slate-700 space-y-2">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Export UE5 .ulens (Zoom)</h3>
            {(!cameraSettings.sensorWidthMm || !cameraSettings.lensName) && (
              <p className="text-[11px] text-yellow-500/80">Set sensor dimensions and lens name in Settings for accurate export.</p>
            )}
            <button type="button" onClick={doExport} disabled={exportStatus === 'loading' || !ws}
              className={`px-5 py-2 rounded-lg text-sm font-semibold transition-colors ${
                exportStatus === 'success' ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' :
                exportStatus === 'error'   ? 'bg-red-500/10 border border-red-500/30 text-red-400' :
                exportStatus === 'loading' ? 'bg-slate-700 text-slate-400 cursor-wait border border-slate-600' :
                                             'bg-blue-600 hover:bg-blue-500 text-white'
              }`}>
              {exportStatus === 'loading' ? 'Exporting…' : exportStatus === 'success' ? 'Exported ✓' : exportStatus === 'error' ? 'Export failed' : 'Export UE5 .ulens (multi-FL)'}
            </button>
            {exportPath && <p className="text-[10px] text-slate-500 font-mono truncate" title={exportPath}>{exportPath}</p>}
          </div>
        </div>

        {/* Right: full ResultPanel for selected FL */}
        {selectedFlResult && !selectedFlResult.error && (
          <ResultPanel
            result={flResultToCalibResult(selectedFlResult, imageSize)}
            imageSize={imageSize}
            ws={ws}
          />
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [connStatus, setConnStatus] = useState<ConnStatus>('connecting');
  const [backendPort, setBackendPort] = useState(8000);
  const [localIp, setLocalIp] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [tab, setTab] = useState<'file' | 'live' | 'results' | 'settings'>('file');

  // Device state (lifted up, shared between Settings and Live Calibration)
  const [devices, setDevices]         = useState<CaptureDevice[]>([]);
  const [scanning, setScanning]       = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [detectedW,   setDetectedW]   = useState(1920);
  const [detectedH,   setDetectedH]   = useState(1080);
  const [detectedFps, setDetectedFps] = useState(30);

  // Lens settings (shared across both tabs)
  const [lensSettings, setLensSettings] = useState<LensSettings>({ lensType: 'spherical', squeezeRatio: 1.0 });

  // Camera / sensor settings (shared across both tabs)
  const [cameraSettings, setCameraSettings] = useState<CameraSettings>({ lensName: '', sensorWidthMm: '', sensorHeightMm: '', nodalPreset: '' });

  // Calibration state (shared across both tabs)
  const [boardSettings, setBoardSettings] = useState<BoardSettings>({ cols: 9, rows: 6, squareSizeMm: 25 });
  const [frames, setFrames] = useState<ScoredFrame[]>([]);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [calibrating, setCalibrating] = useState(false);
  const [calibResult, setCalibResult] = useState<CalibrationResult | null>(null);
  const [calibError, setCalibError] = useState<string | null>(null);
  const [calibImageSize, setCalibImageSize] = useState<[number, number]>([1920, 1080]);
  const [calibFrames, setCalibFrames] = useState<PreviewFrame[]>([]);

  // Zoom calibration result (from GuidedCapture multi-FL path)
  const [zoomResult, setZoomResult] = useState<ZoomCalibResult | null>(null);
  const [zoomImageSize, setZoomImageSize] = useState<[number, number]>([1920, 1080]);
  const [zoomExportStatus, setZoomExportStatus] = useState<'idle'|'loading'|'success'|'error'>('idle');
  const [zoomExportPath, setZoomExportPath] = useState<string | null>(null);
  const [selectedZoomFlMm, setSelectedZoomFlMm] = useState<number | null>(null);

  // Refs hold context set just before the calibrate WS message is sent,
  // so the calibrate_result handler can read the correct imageSize/frames.
  const pendingImageSizeRef = useRef<[number, number]>([1920, 1080]);
  const pendingFramesRef    = useRef<PreviewFrame[]>([]);

  // Connect WebSocket with exponential-backoff reconnection.
  // Attempt delays: 1 s, 2 s, 4 s, 8 s, capped at 16 s.
  const wsPortRef = useRef<number>(8000);
  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    async function connect() {
      if (cancelled) return;
      // Only fetch the port once; after that reuse the cached value
      if (attempt === 0) {
        try {
          const port = window.electronAPI
            ? await window.electronAPI.getBackendPort()
            : 8000;
          wsPortRef.current = port;
          setBackendPort(port);
        } catch {
          setConnStatus('error');
        }
      }
      if (cancelled) return;
      setConnStatus('connecting');
      socket = new WebSocket(`ws://127.0.0.1:${wsPortRef.current}/ws`);

      socket.onopen = () => {
        attempt = 0;
        setConnStatus('connected');
        setWs(socket);
      };

      socket.onclose = () => {
        setConnStatus('disconnected');
        setWs(null);
        if (!cancelled) {
          const delay = Math.min(1000 * 2 ** attempt, 16000);
          attempt += 1;
          retryTimer = setTimeout(connect, delay);
        }
      };

      socket.onerror = () => {
        setConnStatus('error');
        // onclose fires after onerror, which triggers the retry
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  // Scan devices on WS connect, and re-scan whenever live/settings tab opens with empty list
  useEffect(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setScanning(true);
    ws.send(JSON.stringify({ action: 'list_devices' }));
  }, [ws]);

  useEffect(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if ((tab === 'live' || tab === 'settings') && devices.length === 0 && !scanning) {
      setScanning(true);
      ws.send(JSON.stringify({ action: 'list_devices' }));
    }
  }, [tab, ws, devices.length, scanning]);

  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.action === 'device_list') {
          setDevices(msg.devices ?? []);
          setScanning(false);
          const auto = (msg.devices as CaptureDevice[]).find(d => d.brand.is_capture_card)
                    ?? (msg.devices as CaptureDevice[])[0];
          if (auto != null) setSelectedIdx(auto.index);
        } else if (msg.action === 'preview_started' || msg.action === 'live_capture_started') {
          if (msg.actual_width)  setDetectedW(msg.actual_width);
          if (msg.actual_height) setDetectedH(msg.actual_height);
          if (msg.actual_fps)    setDetectedFps(msg.actual_fps);
        } else if (msg.action === 'preview_fps_update' || msg.action === 'live_fps_update') {
          if (msg.actual_fps) setDetectedFps(msg.actual_fps);
        }
      } catch { /* ignore */ }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  const scanDevices = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setScanning(true);
    setDevices([]);
    ws.send(JSON.stringify({ action: 'list_devices' }));
  };

  // Fetch local IP on mount (for network access display)
  useEffect(() => {
    (async () => {
      try {
        if (window.electronAPI?.getLocalIP) {
          const ip = await window.electronAPI.getLocalIP();
          setLocalIp(ip);
        }
      } catch {
        setLocalIp('localhost');
      }
    })();
  }, []);

  // Single calibrate_result listener — works for both file and live tabs
  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.action === 'calibrate_result') {
          setCalibrating(false);
          setTab('results');
          if (msg.error || msg.rms == null) {
            setCalibError(msg.error ?? 'Calibration failed — check your frames and try again.');
            setCalibResult(null);
          } else {
            setCalibError(null);
            setCalibResult(msg as CalibrationResult);
            setCalibImageSize(pendingImageSizeRef.current);
            setCalibFrames(pendingFramesRef.current);
          }
        } else if (msg.action === 'export_result' && msg.format === 'ue5_ulens_zoom') {
          setZoomExportStatus(msg.success ? 'success' : 'error');
          if (msg.success) setZoomExportPath(msg.output_path);
        }
      } catch { /* ignore */ }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  const toggleExclude = (path: string) => {
    setExcluded(prev => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });
  };

  const imageSize: [number, number] = frames[0]
    ? [frames[0].image_width, frames[0].image_height]
    : [1920, 1080];

  const includedFrames = frames.filter(f => !excluded.has(f.path));
  const canCalibrate = includedFrames.filter(f => f.quality !== 'fail').length >= 3;

  const runCalibration = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN || calibrating) return;
    // Store context so the calibrate_result handler picks the right imageSize/frames
    pendingImageSizeRef.current = imageSize;
    pendingFramesRef.current = includedFrames.map((f, i) => ({
      path: f.path, quality: f.quality, sharpness: f.sharpness, coverage: f.coverage, index: i,
    }));
    setCalibrating(true);
    ws.send(JSON.stringify({
      action: 'calibrate',
      scored_frames: includedFrames,
      board_cols: boardSettings.cols,
      board_rows: boardSettings.rows,
      square_size_mm: boardSettings.squareSizeMm,
      image_size: imageSize,
      lens_type: lensSettings.lensType,
      squeeze_ratio: lensSettings.squeezeRatio,
    }));
  };

  // Called by GuidedCapture just before it sends its calibrate message
  const onGuidedCalibrationSent = (imgSize: [number, number], previewFrames: PreviewFrame[]) => {
    pendingImageSizeRef.current = imgSize;
    pendingFramesRef.current    = previewFrames;
  };

  // Called by GuidedCapture when zoom calibration completes
  const onZoomCalibrationComplete = (result: ZoomCalibResult, imgSize: [number, number]) => {
    setZoomResult(result);
    setZoomImageSize(imgSize);
    setZoomExportStatus('idle');
    setZoomExportPath(null);
    setTab('results');
  };

  const TABS = [
    { key: 'file',     label: 'File Calibration' },
    { key: 'live',     label: 'Live Calibration' },
    { key: 'results',  label: 'Results'          },
    { key: 'settings', label: 'Settings'         },
  ] as const;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 flex flex-col">

      {/* Top bar */}
      <header className="flex items-center gap-6 px-6 py-3 border-b border-slate-700/60 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <h1 className="text-base font-semibold tracking-tight">Lens Calibration</h1>
        <nav className="flex gap-1">
          {TABS.map(t => (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              className={`relative px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                tab === t.key
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {t.label}
              {t.key === 'results' && (calibResult || zoomResult) && tab !== 'results' && (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-emerald-400" />
              )}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs">
            <span className={`w-2 h-2 rounded-full ${STATUS_DOT[connStatus] ?? 'bg-slate-500'}`} />
            <span className="text-slate-400">{connStatus}</span>
          </div>
          {localIp && (
            <div className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-700/50 border border-slate-600/50 text-[11px] font-mono text-slate-300">
              <span>📱</span>
              <span>http://{localIp}:5173</span>
            </div>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-5">

        {/* ── FILE CALIBRATION ── */}
        {tab === 'file' && (
          <FileCalibration
            ws={ws}
            boardSettings={boardSettings}
            onBoardChange={setBoardSettings}
            lensSettings={lensSettings}
            cameraSettings={cameraSettings}
            onCameraSettingsChange={setCameraSettings}
            backendPort={backendPort}
            calibrating={calibrating}
            setCalibrating={setCalibrating}
            onZoomCalibrationComplete={onZoomCalibrationComplete}
            onSingleCalibrationSent={(imgSize, previewFrames) => {
              pendingImageSizeRef.current = imgSize;
              pendingFramesRef.current = previewFrames;
            }}
          />
        )}

        {/* ── LIVE CALIBRATION ── */}
        {tab === 'live' && (
          <GuidedCapture
            ws={ws}
            boardSettings={boardSettings}
            backendPort={backendPort}
            onCalibrationSent={onGuidedCalibrationSent}
            onZoomCalibrationComplete={onZoomCalibrationComplete}
            selectedIdx={selectedIdx}
            selectedDeviceName={devices.find(d => d.index === selectedIdx)?.name ?? null}
            detectedW={detectedW}
            detectedH={detectedH}
            detectedFps={detectedFps}
            lensSettings={lensSettings}
            onLensChange={setLensSettings}
            cameraSettings={cameraSettings}
          />
        )}

        {/* ── SETTINGS ── */}
        {tab === 'settings' && (
          <div className="max-w-xl space-y-5">

            {/* Capture Device */}
            <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Capture Device</h2>

              {scanning && (
                <span className="flex items-center gap-2 text-xs text-slate-400">
                  <span className="inline-block w-3 h-3 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                  Scanning…
                </span>
              )}
              {!scanning && devices.length === 0 && (
                <p className="text-xs text-slate-500 italic">No device found</p>
              )}
              {!scanning && devices.length === 1 && (
                <div className="flex items-center gap-3">
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${BRAND_COLOR[devices[0].brand.id] ?? BRAND_COLOR.generic}`}>
                    {devices[0].brand.icon}
                  </span>
                  <span className="text-sm text-slate-200">{devices[0].name}</span>
                  {detectedW > 0 && (
                    <span className="ml-auto text-xs text-slate-500">{detectedW}×{detectedH} · {Math.abs(detectedFps - Math.round(detectedFps)) < 0.01 ? Math.round(detectedFps) : detectedFps.toFixed(2)} fps</span>
                  )}
                </div>
              )}
              {!scanning && devices.length > 1 && (
                <select
                  value={selectedIdx ?? ''}
                  onChange={e => setSelectedIdx(Number(e.target.value))}
                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                >
                  {devices.map(dev => (
                    <option key={dev.index} value={dev.index}>[{dev.brand.icon}] {dev.name}</option>
                  ))}
                </select>
              )}

              <button
                type="button"
                onClick={scanDevices}
                disabled={scanning || !ws}
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-40 text-slate-300 rounded-lg text-xs transition-colors"
              >
                <span>↺</span> Rescan devices
              </button>
            </div>

            {/* Lens Type */}
            <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Lens Type</h2>

              {/* Toggle spherical / anamorphic */}
              <div className="flex gap-2">
                {(['spherical', 'anamorphic'] as const).map(t => (
                  <button key={t} type="button"
                    onClick={() => setLensSettings(s => ({ ...s, lensType: t }))}
                    className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-colors ${
                      lensSettings.lensType === t
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}>{t}</button>
                ))}
              </div>

              {/* Squeeze ratio — only shown when anamorphic */}
              {lensSettings.lensType === 'anamorphic' && (
                <div className="space-y-2">
                  <p className="text-xs text-slate-400">Squeeze Ratio</p>
                  <div className="flex gap-2 flex-wrap">
                    {[1.33, 1.5, 1.8, 2.0].map(r => (
                      <button key={r} type="button"
                        onClick={() => setLensSettings(s => ({ ...s, squeezeRatio: r }))}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          lensSettings.squeezeRatio === r
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}>{r}×</button>
                    ))}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500">Custom:</span>
                    <input type="number" min={1} max={4} step={0.01}
                      value={lensSettings.squeezeRatio}
                      onChange={e => {
                        const v = parseFloat(e.target.value);
                        if (!isNaN(v) && v > 1) setLensSettings(s => ({ ...s, squeezeRatio: v }));
                      }}
                      className="w-20 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500" />
                    <span className="text-xs text-slate-500">×</span>
                  </div>
                </div>
              )}
            </div>

            {/* Camera / Sensor */}
            <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Camera / Sensor</h2>

              <div className="flex flex-col gap-1">
                <span className="text-xs text-slate-400">Camera preset</span>
                <select defaultValue=""
                  onChange={e => {
                    const p = SENSOR_PRESETS.find(p => p.label === e.target.value);
                    if (p) setCameraSettings(s => ({ ...s, sensorWidthMm: String(p.w), sensorHeightMm: String(p.h) }));
                  }}
                  className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 w-full">
                  <option value="" disabled>Select camera…</option>
                  {SENSOR_PRESETS.map(p => <option key={p.label} value={p.label}>{p.label}</option>)}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Sensor width (mm)</span>
                  <input type="number" value={cameraSettings.sensorWidthMm}
                    onChange={e => setCameraSettings(s => ({ ...s, sensorWidthMm: e.target.value }))}
                    placeholder="36.0"
                    className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Sensor height (mm)</span>
                  <input type="number" value={cameraSettings.sensorHeightMm}
                    onChange={e => setCameraSettings(s => ({ ...s, sensorHeightMm: e.target.value }))}
                    placeholder="24.0"
                    className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
                </label>
              </div>

              <label className="flex flex-col gap-1">
                <span className="text-xs text-slate-400">Lens name</span>
                <input value={cameraSettings.lensName}
                  onChange={e => setCameraSettings(s => ({ ...s, lensName: e.target.value }))}
                  placeholder="e.g. Fujinon Premista 28-100"
                  className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-xs text-slate-400">Nodal preset</span>
                <input value={cameraSettings.nodalPreset}
                  onChange={e => setCameraSettings(s => ({ ...s, nodalPreset: e.target.value }))}
                  placeholder="e.g. fujinon-premista-28-100 (leave blank to skip)"
                  className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
                <span className="text-[11px] text-slate-500">Key from nodal_presets.json — overrides OpenCV focal length and nodal offset with manufacturer values.</span>
              </label>
            </div>

            {/* Calibration Chart */}
            <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-3">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Calibration Chart</h2>
              <BoardPresetSelector boardSettings={boardSettings} onBoardChange={setBoardSettings} />
            </div>

          </div>
        )}

        {/* ── RESULTS ── */}
        {tab === 'results' && (
          calibError ? (
            <div className="flex flex-col items-center justify-center h-64 gap-3">
              <span className="text-red-400 text-sm font-semibold">Calibration failed</span>
              <span className="text-slate-400 text-xs text-center max-w-lg">{calibError}</span>
            </div>
          ) : calibResult ? (
            <div className="grid grid-cols-5 gap-5 items-start">
              <div className="col-span-2">
                <ResultPanel result={calibResult} imageSize={calibImageSize} ws={ws} />
              </div>
              <div className="col-span-3">
                <UndistortPreview
                  scoredFrames={calibFrames}
                  cameraMatrix={calibResult.camera_matrix}
                  distCoeffs={calibResult.dist_coeffs}
                  ws={ws}
                />
              </div>
            </div>
          ) : zoomResult ? (
            <ZoomResultsPanel
              result={zoomResult}
              imageSize={zoomImageSize}
              ws={ws}
              lensSettings={lensSettings}
              cameraSettings={cameraSettings}
              exportStatus={zoomExportStatus}
              exportPath={zoomExportPath}
              onExportStatusChange={setZoomExportStatus}
              selectedFlMm={selectedZoomFlMm}
              onSelectFl={setSelectedZoomFlMm}
            />
          ) : (
            <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
              No calibration results yet. Run a calibration from the File or Live Capture tab.
            </div>
          )
        )}
      </main>

      {/* Bottom status bar */}
      <footer className="flex items-center gap-4 px-6 py-2 border-t border-slate-700/60 bg-slate-900 text-xs text-slate-500">
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${STATUS_DOT[connStatus]}`} />
          <span className="capitalize">{connStatus}</span>
        </div>
        <span className="text-slate-700">|</span>
        <span>Port {backendPort}</span>
        {frames.length > 0 && (
          <>
            <span className="text-slate-700">|</span>
            <span>{frames.length} frames scored</span>
          </>
        )}
        {calibResult && (
          <>
            <span className="text-slate-700">|</span>
            <span>RMS {calibResult.rms.toFixed(3)} px · {calibResult.confidence}</span>
          </>
        )}
        {zoomResult && !calibResult && (
          <>
            <span className="text-slate-700">|</span>
            <span>Zoom: {zoomResult.fl_results.length} FLs calibrated</span>
          </>
        )}
      </footer>
    </div>
  );
}
