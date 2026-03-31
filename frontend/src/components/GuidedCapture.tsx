import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, CalibrationResult, ScoredFrame } from '../types';
import BoardPresetSelector from './BoardPresetSelector';
import PoseDiagram from './PoseDiagram';
import ResultPanel from './ResultPanel';

const STREAM_W = 640;
const STREAM_H = 360;
const MIN_GOOD_FOR_CALIB = 5;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DeviceBrand {
  id: string;
  name: string;
  icon: string;
  is_capture_card: boolean;
}

interface CaptureDevice {
  index: number;
  name: string;
  brand: DeviceBrand;
}

interface PoseItem {
  id: string;
  name: string;
  hint: string;
  satisfied: boolean;
}

interface LiveFrame {
  frame: string;
  found: boolean;
  corners: [number, number][];
  quality: string;
  sharpness: number;
  coverage: number;
  auto_captured: boolean;
  frame_count: number;
  checklist: PoseItem[];
  satisfied_count: number;
  total: number;
  complete: boolean;
  next_hint: string;
  next_pose_id: string | null;
  matching_pose_id: string | null;
  hold_progress: number;   // 0–1: how long the board has been held in the matching pose
  message: string;
}

interface Props {
  ws: WebSocket | null;
  boardSettings: BoardSettings;
  onBoardChange: (s: BoardSettings) => void;
  backendPort: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const QUALITY_COLOR: Record<string, string> = {
  good: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10',
  warn: 'text-yellow-400  border-yellow-500/40  bg-yellow-500/10',
  fail: 'text-red-400     border-red-500/40     bg-red-500/10',
};

const BRAND_COLOR: Record<string, string> = {
  blackmagic: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  aja:        'bg-red-500/20    text-red-300    border-red-500/30',
  bluefish444:'bg-blue-500/20   text-blue-300   border-blue-500/30',
  magewell:   'bg-purple-500/20 text-purple-300 border-purple-500/30',
  datapath:   'bg-orange-500/20 text-orange-300 border-orange-500/30',
  deltacast:  'bg-cyan-500/20   text-cyan-300   border-cyan-500/30',
  generic:    'bg-slate-700     text-slate-400  border-slate-600',
};

const RESOLUTIONS = [
  { label: '4K UHD  3840×2160', w: 3840, h: 2160 },
  { label: '4K DCI  4096×2160', w: 4096, h: 2160 },
  { label: '2K DCI  2048×1080', w: 2048, h: 1080 },
  { label: '1080p   1920×1080', w: 1920, h: 1080 },
  { label: '720p    1280×720',  w: 1280, h: 720  },
  { label: '576p    720×576',   w: 720,  h: 576  },
  { label: '480p    720×480',   w: 720,  h: 480  },
];

// fps stored as number (exact float); label is display string
const FRAMERATES: { label: string; fps: number }[] = [
  { label: '23.976',  fps: 24000 / 1001 },
  { label: '24',      fps: 24           },
  { label: '25',      fps: 25           },
  { label: '29.97',   fps: 30000 / 1001 },
  { label: '30',      fps: 30           },
  { label: '47.952',  fps: 48000 / 1001 },
  { label: '48',      fps: 48           },
  { label: '50',      fps: 50           },
  { label: '59.94',   fps: 60000 / 1001 },
  { label: '60',      fps: 60           },
  { label: '119.88',  fps: 120000 / 1001},
  { label: '120',     fps: 120          },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Hold-ring — circular countdown SVG overlaid on the video
// ---------------------------------------------------------------------------

function HoldRing({ progress }: { progress: number }) {
  const R  = 28;           // circle radius
  const cx = 50, cy = 50;  // centre of the 100×100 viewBox
  const circumference = 2 * Math.PI * R;
  const dash = circumference * progress;

  return (
    <svg
      viewBox="0 0 100 100"
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    >
      {/* Background track */}
      <circle cx={cx} cy={cy} r={R}
        fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="4" />
      {/* Progress arc — starts from top (rotated -90°) */}
      <circle cx={cx} cy={cy} r={R}
        fill="none"
        stroke={progress >= 0.99 ? '#10b981' : '#818cf8'}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circumference}`}
        transform={`rotate(-90 ${cx} ${cy})`}
        className="hold-ring-arc"
      />
      {/* Centre label */}
      <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
        fontSize="12" fontWeight="bold" fill="white" opacity="0.9"
      >
        {progress >= 0.99 ? '✓' : `${Math.round(progress * 100)}%`}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------

export default function GuidedCapture({ ws, boardSettings, onBoardChange }: Props) {
  const [devices, setDevices]           = useState<CaptureDevice[]>([]);
  const [scanning, setScanning]         = useState(false);
  const [selectedIdx, setSelectedIdx]   = useState<number | null>(null);
  const [resolution, setResolution]     = useState(RESOLUTIONS[3]); // 1080p default
  const [framerate, setFramerate]       = useState(FRAMERATES[4]);  // 30 default

  const [running, setRunning]           = useState(false);
  const [liveFrame, setLiveFrame]       = useState<LiveFrame | null>(null);
  const [actualSize, setActualSize]     = useState<[number, number]>([1920, 1080]);
  const [flash, setFlash]               = useState(false);
  const [capturedFrames, setCapturedFrames] = useState<ScoredFrame[]>([]);
  const [checklist, setChecklist]       = useState<PoseItem[]>([]);
  const [checklistComplete, setChecklistComplete] = useState(false);
  const [calibrating, setCalibrating]   = useState(false);
  const [calibResult, setCalibResult]   = useState<CalibrationResult | null>(null);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const flashTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Scan for devices on mount
  useEffect(() => {
    if (ws) scanDevices();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ws]);

  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);

        if (msg.action === 'device_list') {
          setDevices(msg.devices ?? []);
          setScanning(false);
          // Auto-select first capture card, or first device
          const first_card = (msg.devices as CaptureDevice[]).find(d => d.brand.is_capture_card);
          const first_any  = (msg.devices as CaptureDevice[])[0];
          const auto = first_card ?? first_any;
          if (auto != null) setSelectedIdx(auto.index);

        } else if (msg.action === 'live_capture_started') {
          setActualSize([msg.actual_width, msg.actual_height]);

        } else if (msg.action === 'live_frame') {
          setLiveFrame(msg as LiveFrame);
          if (msg.checklist) {
            setChecklist(msg.checklist);
            setChecklistComplete(msg.complete ?? false);
          }
          if (msg.auto_captured) {
            setFlash(true);
            if (flashTimer.current) clearTimeout(flashTimer.current);
            flashTimer.current = setTimeout(() => setFlash(false), 500);
          }

        } else if (msg.action === 'live_capture_stopped') {
          setRunning(false);
          setCapturedFrames(msg.scored_frames ?? []);

        } else if (msg.action === 'calibrate_result') {
          setCalibResult(msg);
          setCalibrating(false);
        }
      } catch { /* ignore */ }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  // Draw corner overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !liveFrame) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, STREAM_W, STREAM_H);
    if (!liveFrame.found || !liveFrame.corners.length) return;

    const corners = liveFrame.corners;
    const cols = boardSettings.cols;

    ctx.fillStyle = '#10b981';
    for (const [x, y] of corners) {
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    if (corners.length >= 4) {
      const c0 = corners[0];
      const c1 = corners[Math.min(cols - 1, corners.length - 1)];
      const c2 = corners[corners.length - 1];
      const c3 = corners[Math.max(0, corners.length - cols)];
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(c0[0], c0[1]);
      ctx.lineTo(c1[0], c1[1]);
      ctx.lineTo(c2[0], c2[1]);
      ctx.lineTo(c3[0], c3[1]);
      ctx.closePath();
      ctx.stroke();
    }
  }, [liveFrame, boardSettings.cols]);

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  const scanDevices = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setScanning(true);
    setDevices([]);
    ws.send(JSON.stringify({ action: 'list_devices' }));
  };

  const startCapture = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN || selectedIdx === null) return;
    setCapturedFrames([]);
    setCalibResult(null);
    setLiveFrame(null);
    setChecklist([]);
    setChecklistComplete(false);
    setRunning(true);
    ws.send(JSON.stringify({
      action:     'start_live_capture',
      device:     selectedIdx,
      width:      resolution.w,
      height:     resolution.h,
      fps:        framerate.fps,
      board_cols: boardSettings.cols,
      board_rows: boardSettings.rows,
      save_dir:   'captures',
    }));
  };

  const stopCapture = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: 'stop_live_capture' }));
  };

  const manualCapture = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: 'manual_capture' }));
  };

  const runCalibration = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setCalibrating(true);
    ws.send(JSON.stringify({
      action:         'calibrate',
      scored_frames:  capturedFrames,
      board_cols:     boardSettings.cols,
      board_rows:     boardSettings.rows,
      square_size_mm: boardSettings.squareSizeMm,
      image_size:     actualSize,
    }));
  };

  // ---------------------------------------------------------------------------
  // Derived values
  // ---------------------------------------------------------------------------

  const selectedDevice     = devices.find(d => d.index === selectedIdx) ?? null;
  const frameCount         = liveFrame?.frame_count ?? capturedFrames.length;
  const satisfiedCount     = liveFrame?.satisfied_count ?? checklist.filter(p => p.satisfied).length;
  const totalPoses         = liveFrame?.total ?? 10;
  const progressPct        = Math.min(100, (satisfiedCount / totalPoses) * 100);
  const canCalibrate       = !running && capturedFrames.length >= MIN_GOOD_FOR_CALIB;
  const nextHint           = liveFrame?.next_hint ?? '';
  const displayChecklist   = running ? (liveFrame?.checklist ?? checklist) : checklist;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-4">

      {/* ── Device selection row ─────────────────────────────────────── */}
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Capture Device
          </h3>
          <button
            type="button"
            onClick={scanDevices}
            disabled={scanning || !ws || running}
            className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-700 hover:bg-slate-600 disabled:opacity-40 text-xs text-slate-300 transition-colors"
          >
            {scanning
              ? <span className="inline-block w-3 h-3 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
              : '↺'}
            {scanning ? 'Scanning…' : 'Detect'}
          </button>
        </div>

        {/* Device list */}
        {devices.length === 0 && !scanning && (
          <p className="text-xs text-slate-500 italic">No devices found — press Detect</p>
        )}
        {devices.length > 0 && (
          <div className="space-y-1.5">
            {devices.map(dev => (
              <button
                type="button"
                key={dev.index}
                onClick={() => setSelectedIdx(dev.index)}
                disabled={running}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg border text-left transition-colors ${
                  selectedIdx === dev.index
                    ? 'border-blue-500/60 bg-blue-500/10'
                    : 'border-slate-700 bg-slate-700/50 hover:bg-slate-700'
                } disabled:opacity-60`}
              >
                {/* Brand badge */}
                <span className={`shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded border ${BRAND_COLOR[dev.brand.id] ?? BRAND_COLOR.generic}`}>
                  {dev.brand.icon}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-slate-200 truncate">{dev.name}</p>
                  <p className="text-[10px] text-slate-500">{dev.brand.name} — index {dev.index}</p>
                </div>
                {selectedIdx === dev.index && (
                  <span className="shrink-0 text-blue-400 text-sm">✓</span>
                )}
              </button>
            ))}
          </div>
        )}

        {/* Resolution + board settings row */}
        <div className="flex items-end gap-3 flex-wrap pt-1 border-t border-slate-700">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-slate-500 uppercase tracking-wider">Resolution</span>
            <select
              value={resolution.label}
              onChange={e => setResolution(RESOLUTIONS.find(r => r.label === e.target.value) ?? RESOLUTIONS[3])}
              disabled={running}
              className="bg-slate-700 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              {RESOLUTIONS.map(r => (
                <option key={r.label} value={r.label}>{r.label}</option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-slate-500 uppercase tracking-wider">Frame rate</span>
            <select
              value={framerate.label}
              onChange={e => setFramerate(FRAMERATES.find(f => f.label === e.target.value) ?? FRAMERATES[4])}
              disabled={running}
              className="bg-slate-700 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              {FRAMERATES.map(f => (
                <option key={f.label} value={f.label}>{f.label}</option>
              ))}
            </select>
          </label>

          <div className="flex-1 min-w-[220px]">
            <BoardPresetSelector
              boardSettings={boardSettings}
              onBoardChange={onBoardChange}
              disabled={running}
            />
          </div>

          <div className="flex gap-2 pb-0.5 ml-auto">
            {!running ? (
              <button
                type="button"
                onClick={startCapture}
                disabled={!ws || selectedIdx === null}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Start Capture
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={manualCapture}
                  className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Capture
                </button>
                <button
                  type="button"
                  onClick={stopCapture}
                  className="px-3 py-2 bg-red-600/80 hover:bg-red-600 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Stop
                </button>
              </>
            )}
          </div>
        </div>

        {/* Active device info bar */}
        {running && selectedDevice && (
          <div className="flex items-center gap-2 text-[11px] text-slate-400 pt-1 border-t border-slate-700">
            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${BRAND_COLOR[selectedDevice.brand.id] ?? BRAND_COLOR.generic}`}>
              {selectedDevice.brand.icon}
            </span>
            <span className="truncate">{selectedDevice.name}</span>
            <span className="text-slate-600">·</span>
            <span>{actualSize[0]}×{actualSize[1]} · {framerate.label} fps</span>
            <span className="ml-auto flex items-center gap-1 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
              Live
            </span>
          </div>
        )}
      </div>

      {/* ── Video + checklist ────────────────────────────────────────── */}
      <div className="flex gap-4 items-start flex-wrap lg:flex-nowrap">

        {/* Left: video + hint */}
        <div className="flex-shrink-0 space-y-3 w-full lg:w-[640px]">

          {/* Video — fixed 16:9 aspect ratio matching STREAM_W/STREAM_H (640×360) */}
          <div className="rounded-xl overflow-hidden bg-slate-900 border border-slate-700 relative aspect-video">
            {liveFrame ? (
              <img
                src={`data:image/jpeg;base64,${liveFrame.frame}`}
                alt="Live camera feed"
                className="absolute inset-0 w-full h-full object-fill"
                draggable={false}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-sm">
                {running ? 'Connecting to camera…' : 'Press Start Capture'}
              </div>
            )}

            <canvas
              ref={canvasRef}
              width={STREAM_W}
              height={STREAM_H}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            <div className={`absolute inset-0 bg-white pointer-events-none transition-opacity duration-500 ${flash ? 'opacity-60' : 'opacity-0'}`} />

            {/* Hold-still countdown ring — appears when board is in a matching pose */}
            {liveFrame && liveFrame.hold_progress > 0 && !liveFrame.complete && (
              <HoldRing progress={liveFrame.hold_progress} />
            )}

            {liveFrame?.found && (
              <div className={`absolute top-2 right-2 text-[10px] font-bold px-2 py-1 rounded-lg border capitalize ${QUALITY_COLOR[liveFrame.quality] ?? ''}`}>
                {liveFrame.quality}
              </div>
            )}

            {running && (
              <div className="absolute top-2 left-2 text-[10px] font-bold px-2 py-1 rounded-lg bg-black/60 text-white">
                {frameCount} captured
              </div>
            )}

            {checklistComplete && running && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="bg-emerald-600 text-white rounded-xl px-6 py-4 text-center shadow-2xl">
                  <div className="text-2xl mb-1">All poses captured!</div>
                  <div className="text-sm opacity-90">Press Stop to calibrate</div>
                </div>
              </div>
            )}
          </div>

          {/* Progress */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{satisfiedCount} / {totalPoses} poses</span>
              <span className="text-slate-500">{frameCount} frames</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <progress
                value={progressPct}
                max={100}
                className={`block w-full h-full [appearance:none] [&::-webkit-progress-bar]:bg-transparent [&::-webkit-progress-value]:rounded-full [&::-webkit-progress-value]:transition-all ${
                  checklistComplete
                    ? '[&::-webkit-progress-value]:bg-emerald-500 [&::-moz-progress-bar]:bg-emerald-500'
                    : '[&::-webkit-progress-value]:bg-blue-500  [&::-moz-progress-bar]:bg-blue-500'
                }`}
              />
            </div>
          </div>

          {/* Next hint */}
          {running && nextHint && (
            <div className={`rounded-lg border px-3 py-3 flex items-center gap-4 ${
              checklistComplete
                ? 'bg-emerald-500/10 border-emerald-500/30'
                : 'bg-slate-800 border-slate-700'
            }`}>
              {/* Large pose diagram */}
              {!checklistComplete && liveFrame?.next_pose_id && (
                <PoseDiagram poseId={liveFrame.next_pose_id} size="lg" />
              )}

              <div className="flex-1 min-w-0">
                <p className={`text-sm font-semibold ${checklistComplete ? 'text-emerald-300' : 'text-slate-200'}`}>
                  {checklistComplete ? '✓ All poses captured!' : nextHint}
                </p>
                {liveFrame?.message && (
                  <p className="text-xs text-slate-500 mt-0.5">{liveFrame.message}</p>
                )}
                {checklistComplete && (
                  <p className="text-xs text-slate-400 mt-0.5">Press Stop to run calibration</p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right: pose checklist */}
        {displayChecklist.length > 0 && (
          <div className="flex-1 min-w-[220px]">
            <div className="rounded-xl bg-slate-800 border border-slate-700 p-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Required Poses
              </h3>
              <div className="space-y-2">
                {displayChecklist.map((pose) => (
                  <div
                    key={pose.id}
                    className={`flex items-center gap-3 px-2 py-2 rounded-lg text-sm transition-colors ${
                      pose.satisfied
                        ? 'bg-emerald-500/10 border border-emerald-500/20'
                        : 'bg-slate-700/50 border border-transparent'
                    }`}
                  >
                    {/* Small pose diagram */}
                    <PoseDiagram poseId={pose.id} size="sm" satisfied={pose.satisfied} />

                    <div className="min-w-0 flex-1">
                      <p className={`text-xs font-semibold leading-tight ${pose.satisfied ? 'text-emerald-300' : 'text-slate-300'}`}>
                        {pose.name}
                      </p>
                      {!pose.satisfied && (
                        <p className="text-[10px] text-slate-500 mt-0.5 leading-snug">{pose.hint}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Calibrate button ─────────────────────────────────────────── */}
      {!running && capturedFrames.length > 0 && (
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
              ? `Run Calibration (${capturedFrames.length} frames)`
              : `Need ${MIN_GOOD_FOR_CALIB - capturedFrames.length} more frames`}
          </button>
          {calibrating && (
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          )}
          {checklistComplete && !calibrating && (
            <span className="text-xs text-emerald-400">All 10 poses captured — ready!</span>
          )}
        </div>
      )}

      {/* ── Results ──────────────────────────────────────────────────── */}
      {calibResult && (
        <ResultPanel
          result={calibResult}
          imageSize={actualSize}
          ws={ws}
        />
      )}
    </div>
  );
}
