import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, CalibrationResult, ConnStatus, ScoredFrame } from './types';
import CoverageMap from './components/CoverageMap';
import DropZone from './components/DropZone';
import FrameGrid from './components/FrameGrid';
import GuidedCapture from './components/GuidedCapture';
import ResultPanel from './components/ResultPanel';
import UndistortPreview from './components/UndistortPreview';

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
  const [tab, setTab] = useState<'file' | 'live'>('file');

  // Calibration state
  const [boardSettings, setBoardSettings] = useState<BoardSettings>({ cols: 9, rows: 6, squareSizeMm: 25 });
  const [frames, setFrames] = useState<ScoredFrame[]>([]);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [calibrating, setCalibrating] = useState(false);
  const [calibResult, setCalibResult] = useState<CalibrationResult | null>(null);

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

  // Listen for calibrate_result in App
  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.action === 'calibrate_result') {
          setCalibResult(msg as CalibrationResult);
          setCalibrating(false);
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
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setCalibrating(true);
    ws.send(JSON.stringify({
      action: 'calibrate',
      scored_frames: includedFrames,
      board_cols: boardSettings.cols,
      board_rows: boardSettings.rows,
      square_size_mm: boardSettings.squareSizeMm,
      image_size: imageSize,
    }));
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 flex flex-col">

      {/* Top bar */}
      <header className="flex items-center gap-6 px-6 py-3 border-b border-slate-700/60 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <h1 className="text-base font-semibold tracking-tight">Lens Calibration</h1>
        <nav className="flex gap-1">
          {(['file', 'live'] as const).map(t => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                tab === t
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {t === 'file' ? 'File Calibration' : 'Live Capture'}
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
                    <span className="text-xs text-slate-500">
                      Last RMS: <span className={`font-mono font-bold ${
                        calibResult.rms < 0.3 ? 'text-emerald-400' :
                        calibResult.rms < 1.0 ? 'text-yellow-400' : 'text-red-400'
                      }`}>{calibResult.rms.toFixed(3)}</span>
                    </span>
                  )}
                </div>
              )}

              {calibResult && (
                <>
                  <ResultPanel result={calibResult} imageSize={imageSize} ws={ws} />
                  <UndistortPreview
                    scoredFrames={frames.filter(f => !excluded.has(f.path)).map(f => ({
                      path: f.path,
                      quality: f.quality,
                      sharpness: f.sharpness,
                      coverage: f.coverage,
                      index: f.index,
                    }))}
                    cameraMatrix={calibResult.camera_matrix}
                    distCoeffs={calibResult.dist_coeffs}
                    ws={ws}
                  />
                </>
              )}
            </div>
          </div>
        )}

        {/* ── LIVE CAPTURE ── */}
        {tab === 'live' && (
          <GuidedCapture
            ws={ws}
            boardSettings={boardSettings}
            onBoardChange={setBoardSettings}
            backendPort={backendPort}
          />
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
