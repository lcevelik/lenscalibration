import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, CameraSettings, CalibrationResult, ConnStatus, LensSettings, ScoredFrame } from './types';
import CoverageMap from './components/CoverageMap';
import DropZone from './components/DropZone';
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
  { label: 'Venice / Venice 2 FF (36.0 × 24.0)',    w: 36.0, h: 24.0 },
  { label: 'Venice 2 / Burano FF 8K (35.9 × 24.0)', w: 35.9, h: 24.0 },
  { label: 'Venice 2 / Burano S35 (26.2 × 14.7)',   w: 26.2, h: 14.7 },
  { label: 'ALEXA 35 LF (27.99 × 19.22)',            w: 27.99, h: 19.22 },
  { label: 'ALEXA 35 S35 (26.40 × 14.85)',           w: 26.40, h: 14.85 },
  { label: 'RED V-RAPTOR 8K VV (40.96 × 21.60)',     w: 40.96, h: 21.60 },
  { label: 'RED KOMODO 6K S35 (27.03 × 14.26)',      w: 27.03, h: 14.26 },
];

const STATUS_DOT: Record<ConnStatus, string> = {
  connecting:   'bg-yellow-400 animate-pulse',
  connected:    'bg-emerald-400',
  disconnected: 'bg-slate-500',
  error:        'bg-red-400',
};

export default function App() {
  const [connStatus, setConnStatus] = useState<ConnStatus>('connecting');
  const [backendPort, setBackendPort] = useState(8000);
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
  const [lensSettings, setLensSettings] = useState<LensSettings>({ lensType: 'spherical', squeezeRatio: 2.0 });

  // Camera / sensor settings (shared across both tabs)
  const [cameraSettings, setCameraSettings] = useState<CameraSettings>({ lensName: '', sensorWidthMm: '', sensorHeightMm: '' });

  // Calibration state (shared across both tabs)
  const [boardSettings, setBoardSettings] = useState<BoardSettings>({ cols: 9, rows: 6, squareSizeMm: 25 });
  const [frames, setFrames] = useState<ScoredFrame[]>([]);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [calibrating, setCalibrating] = useState(false);
  const [calibResult, setCalibResult] = useState<CalibrationResult | null>(null);
  const [calibImageSize, setCalibImageSize] = useState<[number, number]>([1920, 1080]);
  const [calibFrames, setCalibFrames] = useState<PreviewFrame[]>([]);

  // Refs hold context set just before the calibrate WS message is sent,
  // so the calibrate_result handler can read the correct imageSize/frames.
  const pendingImageSizeRef = useRef<[number, number]>([1920, 1080]);
  const pendingFramesRef    = useRef<PreviewFrame[]>([]);

  // Connect WebSocket
  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket;

    async function connect() {
      const port = window.electronAPI
        ? await window.electronAPI.getBackendPort()
        : 8000;
      if (cancelled) return;
      setBackendPort(port);
      socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
      socket.onopen    = () => { setConnStatus('connected'); setWs(socket); };
      socket.onclose   = () => setConnStatus('disconnected');
      socket.onerror   = () => setConnStatus('error');
    }

    connect().catch(() => setConnStatus('error'));
    return () => { cancelled = true; socket?.close(); };
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

  // Single calibrate_result listener — works for both file and live tabs
  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.action !== 'calibrate_result') return;
        setCalibResult(msg as CalibrationResult);
        setCalibrating(false);
        setCalibImageSize(pendingImageSizeRef.current);
        setCalibFrames(pendingFramesRef.current);
        setTab('results');
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
              {t.key === 'results' && calibResult && tab !== 'results' && (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-emerald-400" />
              )}
            </button>
          ))}
        </nav>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-5">

        {/* ── FILE CALIBRATION ── */}
        {tab === 'file' && (
          <div className="grid grid-cols-5 gap-5 items-start">

            {/* Left column */}
            <div className="col-span-2 space-y-4">
              <div className="rounded-xl bg-slate-800 border border-slate-700 p-4">
                <DropZone
                  ws={ws}
                  boardSettings={boardSettings}
                  onBoardChange={setBoardSettings}
                  onScoringDone={newFrames =>
                    setFrames(prev => {
                      const existingPaths = new Set(prev.map(f => f.path));
                      const deduped = newFrames.filter(f => !existingPaths.has(f.path));
                      return [...prev, ...deduped];
                    })
                  }
                />
              </div>

              <FrameGrid
                frames={frames}
                excluded={excluded}
                onToggle={toggleExclude}
                backendPort={backendPort}
              />
            </div>

            {/* Right column */}
            <div className="col-span-3 space-y-4">
              <CoverageMap frames={frames} imageSize={imageSize} />

              {/* Run calibration */}
              {frames.length > 0 && (
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={runCalibration}
                    disabled={!canCalibrate || calibrating}
                    className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors ${
                      canCalibrate && !calibrating
                        ? 'bg-blue-600 hover:bg-blue-500 text-white'
                        : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                    }`}
                  >
                    {calibrating
                      ? 'Calibrating…'
                      : canCalibrate
                      ? `Run Calibration (${includedFrames.length} frames)`
                      : 'Need ≥ 3 non-fail frames'}
                  </button>
                  {calibrating && (
                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  )}
                  {calibResult && !calibrating && (
                    <button
                      type="button"
                      onClick={() => setTab('results')}
                      className="text-xs text-emerald-400 hover:text-emerald-300 underline underline-offset-2"
                    >
                      View results →
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── LIVE CALIBRATION ── */}
        {tab === 'live' && (
          <GuidedCapture
            ws={ws}
            boardSettings={boardSettings}
            backendPort={backendPort}
            onCalibrationSent={onGuidedCalibrationSent}
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
          calibResult ? (
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
      </footer>
    </div>
  );
}
