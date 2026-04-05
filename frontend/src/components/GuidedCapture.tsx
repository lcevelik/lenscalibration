import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, CameraSettings, LensSettings, ScoredFrame } from '../types';
import PoseDiagram from './PoseDiagram';

const STREAM_W         = 640;
const STREAM_H         = 360;
const MIN_FRAMES_PER_FL = 5;

const SENSOR_PRESETS = [
  { label: 'Venice — FF 6K (36.0 × 24.0)',                 w: 36.0,  h: 24.0  },
  { label: 'Venice 2 / Burano — FF 8K (35.9 × 24.0)',      w: 35.9,  h: 24.0  },
  { label: 'Venice 2 / Burano — S35 (26.2 × 14.7)',        w: 26.2,  h: 14.7  },
  { label: 'Venice 2 / Burano — S16 (14.6 × 8.2)',         w: 14.6,  h: 8.2   },
  { label: 'HDC-series 2/3" HD B4 (9.59 × 5.39)',          w: 9.59,  h: 5.39  },
  { label: 'HDC-F5500 / HDC-3500 4K 2/3" (9.59 × 5.39)',   w: 9.59,  h: 5.39  },
  { label: 'HDC-5500 / HDW-series 2/3" (9.59 × 5.39)',     w: 9.59,  h: 5.39  },
];
const AUTO_STOP_MIN_FRAMES = 12;
const MANUAL_ONLY_CAPTURE = true;
const SNAP_FLS = [18, 24, 28, 35, 50, 70, 85, 100, 135, 150, 200];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PoseItem     { id: string; name: string; hint: string; satisfied: boolean; }
interface NextPoseReqs { regions: number[] | null; tilt_min: number; tilt_max: number; size_min: number; }
interface MatchStatus  {
  position_ok: boolean;
  position_hint?: string | null;
  tilt_ok: boolean;
  size_ok: boolean;
  tilt_score?: number;
  tilt_hint?: string;
  size_score?: number;
  size_hint?: string;
}

interface LiveFrame {
  frame: string; found: boolean; corners: [number, number][];
  detection_type?: 'checkerboard' | 'charuco' | 'aruco_grid' | null;
  quality: string; sharpness: number; coverage: number;
  auto_captured: boolean; frame_count: number;
  checklist: PoseItem[]; satisfied_count: number; total: number; complete: boolean;
  next_hint: string; next_pose_id: string | null; matching_pose_id: string | null;
  hold_progress: number; message: string;
  next_pose_reqs: NextPoseReqs | null; match_status: MatchStatus | null;
}

interface FlGroup {
  fl_mm: number;
  frames: ScoredFrame[];
  status: 'pending' | 'capturing' | 'done';
  /** Physical camera-to-chart distance used during capture (mm). 0 = unset. */
  working_distance_mm: number;
}

export interface ZoomFlResult {
  focal_length_mm: number; rms: number | null;
  fx_px: number; fy_px: number; cx_px: number; cy_px: number;
  focal_length_computed_mm: number; dist_coeffs: number[];
  camera_matrix: number[][]; optical_center_world: [number, number, number];
  used_frames: number; per_image_errors: Array<{ path: string; error: number; outlier: boolean }>;
  confidence: string; error?: string | null;
}

export interface ZoomFlInterpolated {
  focal_length_mm: number;
  fx_px: number; fy_px: number; cx_px: number; cy_px: number;
  dist_coeffs: number[];
  camera_matrix: number[][];
  nodal_offset_z_mm: number;
  interpolated: true;
}

export interface ZoomCalibResult {
  success: boolean; error: string | null;
  fl_results: ZoomFlResult[]; nodal_offsets_mm: Record<string, number>;
  fl_interpolated?: ZoomFlInterpolated[];
  image_size?: [number, number];
}

interface Props {
  ws: WebSocket | null;
  boardSettings: BoardSettings;
  backendPort: number;
  onCalibrationSent: (imageSize: [number, number], frames: Array<{
    path: string; quality: 'good'|'warn'|'fail'; sharpness: number; coverage: number; index: number;
  }>) => void;
  onZoomCalibrationComplete?: (result: ZoomCalibResult, imageSize: [number, number]) => void;
  // Device props lifted from App/Settings
  selectedIdx: number | null;
  selectedDeviceName: string | null;
  detectedW: number;
  detectedH: number;
  detectedFps: number;
  lensSettings: LensSettings;
  onLensChange: (s: LensSettings) => void;
  cameraSettings: CameraSettings;
  onCameraSettingsChange: (s: CameraSettings) => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const QUALITY_COLOR: Record<string, string> = {
  good: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10',
  warn: 'text-yellow-400  border-yellow-500/40  bg-yellow-500/10',
  fail: 'text-red-400     border-red-500/40     bg-red-500/10',
};

const CONFIDENCE_COLOR: Record<string, string> = {
  excellent: 'text-emerald-400',
  good:      'text-blue-400',
  marginal:  'text-yellow-400',
  poor:      'text-red-400',
};


function fmtFps(fps: number): string {
  if (fps <= 0) return '—';
  // Use tolerance instead of exact modulo to handle floats like 29.9999…
  const rounded = Math.round(fps);
  return Math.abs(fps - rounded) < 0.01 ? String(rounded) : fps.toFixed(2).replace(/\.?0+$/, '');
}

type DiversityStats = {
  score: number;
  level: 'poor' | 'fair' | 'good';
  sampleCount: number;
  centerSpread: number;
  tiltSpreadDeg: number;
  coverageSpread: number;
};

function computeFrameDiversity(frames: ScoredFrame[]): DiversityStats | null {
  const usable = frames.filter(f => f.quality !== 'fail' && f.corners.length > 0);
  if (usable.length < 3) return null;

  const centers = usable.map(f => {
    const xs = f.corners.map(c => c[0]);
    const ys = f.corners.map(c => c[1]);
    const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
    const w = Math.max(1, f.image_width || 1);
    const h = Math.max(1, f.image_height || 1);
    return [cx / w, cy / h] as const;
  });

  const centerXs = centers.map(c => c[0]);
  const centerYs = centers.map(c => c[1]);
  const spreadX = Math.max(...centerXs) - Math.min(...centerXs);
  const spreadY = Math.max(...centerYs) - Math.min(...centerYs);
  const centerSpread = Math.hypot(spreadX, spreadY);

  const angleVals = usable
    .map(f => f.angle)
    .filter((a): a is number => typeof a === 'number')
    .map(a => Math.abs(a));
  const tiltSpreadDeg = angleVals.length >= 2 ? Math.max(...angleVals) - Math.min(...angleVals) : 0;

  const coverageVals = usable.map(f => Number.isFinite(f.coverage) ? f.coverage : 0);
  const coverageSpread = coverageVals.length >= 2 ? Math.max(...coverageVals) - Math.min(...coverageVals) : 0;

  const centerScore = Math.min(1, centerSpread / 0.35);
  const tiltScore = Math.min(1, tiltSpreadDeg / 12.0);
  const coverageScore = Math.min(1, coverageSpread / 0.08);
  const score = Math.round(100 * (0.55 * centerScore + 0.25 * tiltScore + 0.20 * coverageScore));

  let level: DiversityStats['level'] = 'poor';
  if (score >= 70) level = 'good';
  else if (score >= 45) level = 'fair';

  return {
    score,
    level,
    sampleCount: usable.length,
    centerSpread,
    tiltSpreadDeg,
    coverageSpread,
  };
}

// ---------------------------------------------------------------------------
// Hold-ring SVG
// ---------------------------------------------------------------------------

function HoldRing({ progress }: { progress: number }) {
  const R = 28, cx = 50, cy = 50;
  const circumference = 2 * Math.PI * R;
  return (
    <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden="true">
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="4" />
      <circle cx={cx} cy={cy} r={R} fill="none"
        stroke={progress >= 0.99 ? '#10b981' : '#818cf8'} strokeWidth="4" strokeLinecap="round"
        strokeDasharray={`${circumference * progress} ${circumference}`}
        transform={`rotate(-90 ${cx} ${cy})`} />
      <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
        fontSize="12" fontWeight="bold" fill="white" opacity="0.9">
        {progress >= 0.99 ? '✓' : `${Math.round(progress * 100)}%`}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GuidedCapture({ ws, boardSettings, onCalibrationSent, onZoomCalibrationComplete, selectedIdx, selectedDeviceName, detectedW, detectedH, detectedFps, lensSettings, onLensChange, cameraSettings, onCameraSettingsChange }: Props) {
  // ── Preview ───────────────────────────────────────────────────────────────
  const [previewFrame, setPreviewFrame] = useState<string | null>(null);
  const [previewing, setPreviewing]   = useState(false);

  // ── Capture ───────────────────────────────────────────────────────────────
  const [running, setRunning]               = useState(false);
  const [liveFrame, setLiveFrame]           = useState<LiveFrame | null>(null);
  const [actualSize, setActualSize]         = useState<[number, number]>([1920, 1080]);
  const [flash, setFlash]                   = useState(false);
  const [capturedFrames, setCapturedFrames] = useState<ScoredFrame[]>([]);
  const [checklist, setChecklist]           = useState<PoseItem[]>([]);
  const [checklistComplete, setChecklistComplete] = useState(false);
  const [pinnedPoseId, setPinnedPoseId]     = useState<string | null>(null);
  const [calibrating, setCalibrating]       = useState(false);

  // ── FL / Zoom ─────────────────────────────────────────────────────────────
  const [fixedMount, setFixedMount]     = useState(false);
  const [flInput, setFlInput]           = useState('');
  const [flGroups, setFlGroups]         = useState<FlGroup[]>([]);
  const [activeFlIdx, setActiveFlIdx]   = useState<number | null>(null);
  const [zoomResult, setZoomResult]     = useState<ZoomCalibResult | null>(null);
  const [calibratingZoom, setCalibratingZoom] = useState(false);
  const [handoffStatus, setHandoffStatus] = useState('');
  // Zoom calibration mode: how many control points to capture
  type ZoomMode = '2pt' | '3pt' | 'manual';
  const [zoomMode, setZoomMode] = useState<ZoomMode>('2pt');
  const [zoomMinInput, setZoomMinInput] = useState('');
  const [zoomMaxInput, setZoomMaxInput] = useState('');
  const [zoomMidInput, setZoomMidInput] = useState('');
  // ── Zoom export ──────────────────────────────────────────────────────────
  const [exportStatus, setExportStatus]       = useState<'idle'|'loading'|'success'|'error'>('idle');
  const [exportPath, setExportPath]           = useState('');

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const flashTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const flGroupsRef = useRef<FlGroup[]>([]);
  const activeFlIdxRef = useRef<number | null>(null);
  const actualSizeRef = useRef<[number, number]>([1920, 1080]);
  const boardSettingsRef = useRef(boardSettings);
  const lensSettingsRef = useRef(lensSettings);
  const cameraSettingsRef = useRef(cameraSettings);
  const checklistCompleteRef = useRef(false);
  const autoStopRequestedRef = useRef(false);

  useEffect(() => {
    wsRef.current = ws;
  }, [ws]);

  useEffect(() => {
    flGroupsRef.current = flGroups;
  }, [flGroups]);

  useEffect(() => {
    activeFlIdxRef.current = activeFlIdx;
  }, [activeFlIdx]);

  useEffect(() => {
    actualSizeRef.current = actualSize;
  }, [actualSize]);

  useEffect(() => {
    boardSettingsRef.current = boardSettings;
  }, [boardSettings]);

  useEffect(() => {
    lensSettingsRef.current = lensSettings;
  }, [lensSettings]);

  useEffect(() => {
    cameraSettingsRef.current = cameraSettings;
  }, [cameraSettings]);

  useEffect(() => {
    checklistCompleteRef.current = checklistComplete;
  }, [checklistComplete]);

  // Reset calibrating flag if the websocket disconnects mid-calibration
  useEffect(() => {
    if (!ws) setCalibrating(false);
  }, [ws]);

  // ── Auto-start preview ────────────────────────────────────────────────────
  useEffect(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN || running || selectedIdx === null) return;
    setPreviewing(true);
    setPreviewFrame(null);
    ws.send(JSON.stringify({ action: 'start_preview', device: selectedIdx }));
    return () => {
      setPreviewing(false);
      setPreviewFrame(null);
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'stop_preview' }));
    };
  }, [selectedIdx, ws, running]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── WebSocket messages ────────────────────────────────────────────────────
  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.action === 'preview_frame') {
          setPreviewFrame(msg.frame);
        } else if (msg.action === 'preview_stopped') {
          setPreviewing(false); setPreviewFrame(null);
        } else if (msg.action === 'live_capture_started') {
          setActualSize([msg.actual_width, msg.actual_height]);
          setHandoffStatus('');
        } else if (msg.action === 'live_frame') {
          setLiveFrame(msg as LiveFrame);
          if (msg.checklist) {
            setChecklist(msg.checklist);
            setChecklistComplete(msg.complete ?? false);
            // Clear pinned pose if it just became satisfied
            setPinnedPoseId(prev => {
              if (prev && msg.checklist.find((p: PoseItem) => p.id === prev)?.satisfied) return null;
              return prev;
            });
          }
          if (msg.auto_captured) {
            setFlash(true);
            if (flashTimer.current) clearTimeout(flashTimer.current);
            flashTimer.current = setTimeout(() => setFlash(false), 500);
          }
          if (
            !MANUAL_ONLY_CAPTURE
            &&
            msg.complete
            && (msg.frame_count ?? 0) >= AUTO_STOP_MIN_FRAMES
            && !autoStopRequestedRef.current
            && wsRef.current?.readyState === WebSocket.OPEN
          ) {
            autoStopRequestedRef.current = true;
            wsRef.current.send(JSON.stringify({ action: 'stop_live_capture' }));
          }
        } else if (msg.action === 'live_capture_stopped') {
          setRunning(false);
          const frames: ScoredFrame[] = msg.scored_frames ?? [];
          const socket = wsRef.current;
          const currentGroups = flGroupsRef.current;
          const currentActiveFlIdx = activeFlIdxRef.current;
          const imageSize = actualSizeRef.current;
          const currentBoardSettings = boardSettingsRef.current;
          const currentLensSettings = lensSettingsRef.current;
          const currentCameraSettings = cameraSettingsRef.current;
          setCapturedFrames(frames);
          const updatedGroups = currentActiveFlIdx === null
            ? currentGroups
            : currentGroups.map((g, i) =>
                i === currentActiveFlIdx
                  ? {
                      ...g,
                      frames,
                      status: (frames.length > 0 ? 'done' : 'pending') as FlGroup['status'],
                    }
                  : g
              );
          flGroupsRef.current = updatedGroups;
          setFlGroups(updatedGroups);
          setLiveFrame(null);

          const readyGroups = updatedGroups.filter(
            g => g.status === 'done' && g.frames.filter(f => f.quality !== 'fail').length >= MIN_FRAMES_PER_FL,
          );
          const captureCompleted = autoStopRequestedRef.current || checklistCompleteRef.current;
          const nextPendingIdx = currentActiveFlIdx === null
            ? -1
            : updatedGroups.findIndex((g, i) => i > currentActiveFlIdx && g.status !== 'done');
          const canSend = socket?.readyState === WebSocket.OPEN;
          const shouldAutoCalibrateSingle = captureCompleted
            && updatedGroups.length === 1
            && readyGroups.length === 1
            && canSend;
          const shouldAutoCalibrateZoom = captureCompleted
            && updatedGroups.length >= 2
            && readyGroups.length === updatedGroups.length
            && readyGroups.length >= 2
            && canSend;

          if (shouldAutoCalibrateSingle) {
            setHandoffStatus('Starting calibration...');
            onCalibrationSent(
              imageSize,
              frames.map((f, i) => ({ path: f.path, quality: f.quality, sharpness: f.sharpness, coverage: f.coverage, index: i })),
            );
            setCalibrating(true);
            socket.send(JSON.stringify({
              action: 'calibrate', scored_frames: frames,
              board_cols: currentBoardSettings.cols, board_rows: currentBoardSettings.rows,
              square_size_mm: currentBoardSettings.squareSizeMm, image_size: imageSize,
              lens_type: currentLensSettings.lensType,
              squeeze_ratio: currentLensSettings.squeezeRatio,
            }));
          } else if (shouldAutoCalibrateZoom) {
            setHandoffStatus('Starting zoom calibration...');
            setCalibratingZoom(true);
            socket.send(JSON.stringify({
              action: 'calibrate_zoom',
              fl_groups: readyGroups.map(g => ({ focal_length_mm: g.fl_mm, frames: g.frames })),
              board_cols: currentBoardSettings.cols, board_rows: currentBoardSettings.rows,
              square_size_mm: currentBoardSettings.squareSizeMm, image_size: imageSize,
              sensor_width_mm: parseFloat(currentCameraSettings.sensorWidthMm) || 0,
              sensor_height_mm: parseFloat(currentCameraSettings.sensorHeightMm) || 0,
              lens_type: currentLensSettings.lensType,
              squeeze_ratio: currentLensSettings.squeezeRatio,
            }));
          } else if (nextPendingIdx >= 0) {
            setHandoffStatus(`Waiting for ${updatedGroups[nextPendingIdx].fl_mm}mm capture before zoom calibration.`);
            setActiveFlIdx(nextPendingIdx);
          } else if (captureCompleted && updatedGroups.length === 1 && readyGroups.length === 0) {
            // Show frame quality breakdown
            const usableCount = frames.filter(f => f.quality !== 'fail').length;
            const failCount = frames.filter(f => f.quality === 'fail').length;
            const hint = failCount > 0 ? ` (${usableCount} usable, ${failCount} rejected due to blur/focus). Try moving slower or hold the camera more steady.` : '';
            setHandoffStatus(`Calibration not started: need at least ${MIN_FRAMES_PER_FL} usable frames. Got ${usableCount}.${hint}`);
          } else if (captureCompleted && updatedGroups.length >= 2 && readyGroups.length < updatedGroups.length) {
            const shortFls = updatedGroups
              .filter(g => g.status === 'done' && g.frames.filter(f => f.quality !== 'fail').length < MIN_FRAMES_PER_FL)
              .map(g => {
                const usable = g.frames.filter(f => f.quality !== 'fail').length;
                return `${g.fl_mm}mm (${usable}/${MIN_FRAMES_PER_FL} frames)`;
              });
            setHandoffStatus(`Zoom calibration not started: ${shortFls.join(', ')} need more usable frames. Re-capture those focal lengths.`);
          } else if (!canSend) {
            setHandoffStatus('Calibration not started: backend connection is not ready.');
          }
          autoStopRequestedRef.current = false;
        } else if (msg.action === 'calibrate_result') {
          setCalibrating(false);
          setHandoffStatus('');
        } else if (msg.action === 'zoom_calibrate_result') {
          const zr = msg as ZoomCalibResult;
          setZoomResult(zr);
          setCalibratingZoom(false);
          setHandoffStatus('');
          if (onZoomCalibrationComplete) onZoomCalibrationComplete(zr, actualSizeRef.current ?? [1920, 1080]);
        } else if (msg.action === 'export_result' && msg.format === 'ue5_ulens_zoom') {
          setExportStatus(msg.success ? 'success' : 'error');
          if (msg.success) setExportPath(msg.output_path);
        }
      } catch { /* ignore */ }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  // ── Canvas overlay ────────────────────────────────────────────────────────
  // Store the latest liveFrame in a ref so the RAF callback always reads the
  // most recent data without being re-scheduled on every state update.
  const pendingFrameRef = useRef<typeof liveFrame>(null);
  const rafIdRef        = useRef<number | null>(null);

  useEffect(() => {
    pendingFrameRef.current = liveFrame;
    if (rafIdRef.current === null) {
      rafIdRef.current = requestAnimationFrame(() => {
        rafIdRef.current = null;
        const canvas = canvasRef.current;
        const frame  = pendingFrameRef.current;
        if (!canvas || !frame) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, STREAM_W, STREAM_H);
        const reqs = frame.next_pose_reqs;
        if (reqs && !frame.complete) {
          const cw = STREAM_W / 3, ch = STREAM_H / 3;
          const isMatching = frame.hold_progress > 0;
          let x1: number, y1: number, x2: number, y2: number;
          if (reqs.regions === null) {
            x1 = 4; y1 = 4; x2 = STREAM_W - 4; y2 = STREAM_H - 4;
          } else {
            const gxs = reqs.regions.map((r: number) => (r % 3) * cw);
            const gys = reqs.regions.map((r: number) => Math.floor(r / 3) * ch);
            x1 = Math.max(2, Math.min(...gxs));
            y1 = Math.max(2, Math.min(...gys));
            x2 = Math.min(STREAM_W - 2, Math.max(...gxs) + cw);
            y2 = Math.min(STREAM_H - 2, Math.max(...gys) + ch);
          }
          ctx.save();
          ctx.lineWidth = 3;
          ctx.setLineDash([8, 5]);
          ctx.strokeStyle = isMatching ? 'rgba(16,185,129,1.0)' : 'rgba(251,191,36,0.95)';
          ctx.fillStyle   = isMatching ? 'rgba(16,185,129,0.18)' : 'rgba(251,191,36,0.10)';
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          // Corner accents for easier visibility
          ctx.setLineDash([]);
          ctx.lineWidth = 4;
          const accentLen = Math.min(20, (x2 - x1) * 0.2, (y2 - y1) * 0.2);
          for (const [ax, ay, dx, dy] of [[x1,y1,1,1],[x2,y1,-1,1],[x1,y2,1,-1],[x2,y2,-1,-1]] as [number,number,number,number][]) {
            ctx.beginPath(); ctx.moveTo(ax + dx*accentLen, ay); ctx.lineTo(ax, ay); ctx.lineTo(ax, ay + dy*accentLen); ctx.stroke();
          }
          ctx.restore();
        }
        if (!frame.found || !frame.corners.length) return;
        const corners = frame.corners;
        const cols = boardSettings.cols;
        ctx.fillStyle = '#10b981';
        for (const [x, y] of corners) { ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill(); }
        if (corners.length >= 4) {
          ctx.strokeStyle = '#06b6d4'; ctx.lineWidth = 2; ctx.setLineDash([]);
          if ((frame.detection_type ?? 'checkerboard') === 'checkerboard') {
            const c0 = corners[0], c1 = corners[Math.min(cols - 1, corners.length - 1)];
            const c2 = corners[corners.length - 1], c3 = corners[Math.max(0, corners.length - cols)];
            ctx.beginPath(); ctx.moveTo(c0[0], c0[1]); ctx.lineTo(c1[0], c1[1]);
            ctx.lineTo(c2[0], c2[1]); ctx.lineTo(c3[0], c3[1]); ctx.closePath(); ctx.stroke();
          } else {
            // ArUco/partial detections do not provide checkerboard ordering;
            // draw a stable bounding box instead of connecting point indices.
            const xs = corners.map(c => c[0]);
            const ys = corners.map(c => c[1]);
            const x1 = Math.max(0, Math.min(...xs));
            const y1 = Math.max(0, Math.min(...ys));
            const x2 = Math.min(STREAM_W, Math.max(...xs));
            const y2 = Math.min(STREAM_H, Math.max(...ys));
            ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
          }
        }
      });
    }
  }, [liveFrame, boardSettings.cols]);

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  const addFl = (fl: number) => {
    if (isNaN(fl) || fl <= 0 || flGroups.some(g => g.fl_mm === fl)) return;
    const updated = [...flGroups, { fl_mm: fl, frames: [], status: 'pending' as const, working_distance_mm: 0 }].sort((a, b) => a.fl_mm - b.fl_mm);
    setFlGroups(updated);
    // Auto-select the new FL if nothing is currently selected
    if (activeFlIdx === null) {
      setActiveFlIdx(updated.findIndex(g => g.fl_mm === fl));
    }
  };

  const removeFl = (fl_mm: number) => {
    if (running) return;
    setFlGroups(prev => prev.filter(g => g.fl_mm !== fl_mm));
    setActiveFlIdx(null);
  };

  const setFlWorkingDistance = (fl_mm: number, value: string) => {
    const parsed = parseFloat(value);
    const nextMm = Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
    setFlGroups(prev => prev.map(g => (g.fl_mm === fl_mm ? { ...g, working_distance_mm: nextMm } : g)));
  };

  const startCapture = () => {
    if (running || !ws || ws.readyState !== WebSocket.OPEN || selectedIdx === null) return;
    if (activeFlIdx === null) return;
    setCapturedFrames([]); setLiveFrame(null);
    setChecklist([]); setChecklistComplete(false); setPinnedPoseId(null);
    autoStopRequestedRef.current = false;
    setRunning(true);
    const saveDir = `captures/zoom_${flGroups[activeFlIdx].fl_mm}mm`;
    ws.send(JSON.stringify({
      action: 'start_live_capture', device: selectedIdx,
      width: detectedW, height: detectedH, fps: detectedFps,
      board_cols: boardSettings.cols, board_rows: boardSettings.rows,
      save_dir: saveDir,
      focal_length_mm: flGroups[activeFlIdx].fl_mm,
      fixed_mount: fixedMount,
      manual_only: MANUAL_ONLY_CAPTURE,
      lens_type: lensSettings.lensType,
      squeeze_ratio: lensSettings.squeezeRatio,
    }));
    setFlGroups(prev => prev.map((g, i) => i === activeFlIdx ? { ...g, status: 'capturing' } : g));
  };

  const stopCapture = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: 'stop_live_capture' }));
  };

  const manualCapture = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: 'manual_capture' }));
  };

  const selectPose = (poseId: string) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const next = pinnedPoseId === poseId ? null : poseId;
    setPinnedPoseId(next);
    ws.send(JSON.stringify({ action: 'set_target_pose', pose_id: next }));
  };

  const runCalibration = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN || activeFlIdx === null) return;
    const frames = flGroups[activeFlIdx].frames;
    onCalibrationSent(
      actualSize,
      frames.map((f, i) => ({ path: f.path, quality: f.quality, sharpness: f.sharpness, coverage: f.coverage, index: i })),
    );
    setCalibrating(true);
    ws.send(JSON.stringify({
      action: 'calibrate', scored_frames: frames,
      board_cols: boardSettings.cols, board_rows: boardSettings.rows,
      square_size_mm: boardSettings.squareSizeMm, image_size: actualSize,
      lens_type: lensSettings.lensType,
      squeeze_ratio: lensSettings.squeezeRatio,
    }));
  };

  const runZoomCalibration = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const readyGroups = flGroups.filter(g => g.status === 'done' && g.frames.filter(f => f.quality !== 'fail').length >= MIN_FRAMES_PER_FL);
    if (readyGroups.length < 2) return;
    setCalibratingZoom(true);
    ws.send(JSON.stringify({
      action: 'calibrate_zoom',
      fl_groups: readyGroups.map(g => ({ focal_length_mm: g.fl_mm, frames: g.frames, working_distance_mm: g.working_distance_mm || 0 })),
      board_cols: boardSettings.cols, board_rows: boardSettings.rows,
      square_size_mm: boardSettings.squareSizeMm, image_size: actualSize,
      sensor_width_mm: parseFloat(cameraSettings.sensorWidthMm) || 0,
      sensor_height_mm: parseFloat(cameraSettings.sensorHeightMm) || 0,
      lens_type: lensSettings.lensType,
      squeeze_ratio: lensSettings.squeezeRatio,
    }));
  };

  const doZoomExport = async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN || !zoomResult) return;
    let outputPath = 'calibration_zoom.ulens';
    if (window.electronAPI?.showSaveDialog) {
      const dlg = await window.electronAPI.showSaveDialog({
        defaultPath: outputPath,
        filters: [{ name: 'UE5 Lens File', extensions: ['ulens'] }, { name: 'All Files', extensions: ['*'] }],
      });
      if (dlg.canceled || !dlg.filePath) return;
      outputPath = dlg.filePath;
    }
    setExportStatus('loading');
    ws.send(JSON.stringify({
      action: 'export', format: 'ue5_ulens_zoom', output_path: outputPath,
      fl_results: zoomResult.fl_results, nodal_offsets_mm: zoomResult.nodal_offsets_mm,
      image_size: actualSize, lens_name: cameraSettings.lensName.trim() || 'Lens',
      sensor_width_mm: parseFloat(cameraSettings.sensorWidthMm) || 0,
      sensor_height_mm: parseFloat(cameraSettings.sensorHeightMm) || 0,
      nodal_preset: cameraSettings.nodalPreset.trim(),
      lens_type: lensSettings.lensType,
      squeeze_ratio: lensSettings.squeezeRatio,
    }));
  };

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const selectedFlFrameCount = activeFlIdx !== null
    ? (flGroups[activeFlIdx]?.frames.length ?? 0)
    : capturedFrames.length;
  const frameCount       = running ? (liveFrame?.frame_count ?? 0) : selectedFlFrameCount;
  const satisfiedCount   = liveFrame?.satisfied_count ?? checklist.filter(p => p.satisfied).length;
  const totalPoses       = liveFrame?.total ?? 10;
  const progressPct      = Math.min(100, (satisfiedCount / totalPoses) * 100);
  const nextHint         = liveFrame?.next_hint ?? '';
  const readyFlCount     = flGroups.filter(g => g.status === 'done' && g.frames.filter(f => f.quality !== 'fail').length >= MIN_FRAMES_PER_FL).length;
  const canCalibrateZ    = !running && !calibratingZoom && readyFlCount >= 2;

  const nextPendingFlIdx = flGroups.findIndex((g, i) => activeFlIdx !== null && i > activeFlIdx && g.status !== 'done');
  const activeFlFrames = activeFlIdx !== null ? (flGroups[activeFlIdx]?.frames ?? []) : [];
  const activeFlDiversity = computeFrameDiversity(activeFlFrames);

  // Current FL context label
  const currentFl = activeFlIdx !== null ? flGroups[activeFlIdx]?.fl_mm : null;

  // Single-FL calibration: exactly 1 FL is done with enough frames
  const singleFlReady = activeFlIdx !== null
    && flGroups[activeFlIdx]?.status === 'done'
    && flGroups[activeFlIdx].frames.filter(f => f.quality !== 'fail').length >= MIN_FRAMES_PER_FL
    && readyFlCount === 1;
  const canCalibrateSingle = singleFlReady && !running && !calibrating;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-4">

      {/* ── Capture controls row ─────────────────────────────────────── */}
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-4">
        <div className="flex items-center gap-3 flex-wrap">
          {selectedDeviceName && (
            <span className="text-xs text-slate-300 truncate max-w-[200px]" title={selectedDeviceName}>{selectedDeviceName}</span>
          )}
          {!selectedDeviceName && selectedIdx === null && (
            <span className="text-xs text-slate-500 italic">No device — go to Settings</span>
          )}
          {lensSettings.lensType === 'anamorphic' ? (
            <span className="flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-lg border border-indigo-500/50 bg-indigo-500/15 text-indigo-300">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 inline-block" />
              Anamorphic · {lensSettings.squeezeRatio}×
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-lg border border-slate-600 bg-slate-700/60 text-slate-400">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500 inline-block" />
              Spherical
            </span>
          )}
          {(previewFrame || running) && (
            <span className="text-xs text-slate-400">{detectedW}×{detectedH} · {fmtFps(detectedFps)} fps</span>
          )}
          {running && (
            <span className="flex items-center gap-1 text-xs text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />Live
              {currentFl && <span className="text-blue-300 font-semibold ml-1">{currentFl}mm</span>}
            </span>
          )}
          <div className="flex gap-2 ml-auto">
            {!running ? (
              <>
                <button type="button" onClick={startCapture}
                  disabled={!ws || selectedIdx === null || activeFlIdx === null}
                  title={activeFlIdx === null ? 'Add a focal length first' : undefined}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg text-sm font-medium transition-colors">
                  {activeFlIdx !== null
                    ? `Capture at ${flGroups[activeFlIdx]?.fl_mm}mm`
                    : 'Add a focal length first'}
                </button>
              </>
            ) : (
              <>
                <button type="button" onClick={manualCapture}
                  className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors">Capture</button>
                <button type="button" onClick={stopCapture}
                  className="px-3 py-2 bg-red-600/80 hover:bg-red-600 text-white rounded-lg text-sm font-medium transition-colors">Stop</button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── Camera / Sensor ─────────────────────────────────────────── */}
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-4 space-y-3">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Camera / Sensor</h2>
        <select
          aria-label="Camera sensor preset"
          disabled={running}
          value={SENSOR_PRESETS.find(
            p => cameraSettings.sensorWidthMm === String(p.w) && cameraSettings.sensorHeightMm === String(p.h)
          )?.label ?? ''}
          onChange={e => {
            const p = SENSOR_PRESETS.find(p => p.label === e.target.value);
            if (p) onCameraSettingsChange({ ...cameraSettings, sensorWidthMm: String(p.w), sensorHeightMm: String(p.h) });
          }}
          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
        >
          <option value="" disabled>Select camera preset…</option>
          {SENSOR_PRESETS.map(p => (
            <option key={p.label} value={p.label}>{p.label}</option>
          ))}
        </select>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400 shrink-0">Sensor W</span>
            <input type="number" min={1} max={200} step={0.01}
              disabled={running}
              value={cameraSettings.sensorWidthMm}
              onChange={e => onCameraSettingsChange({ ...cameraSettings, sensorWidthMm: e.target.value })}
              placeholder="36.0"
              className="w-20 px-2 py-1 rounded-md border border-slate-600 text-sm text-slate-200 bg-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50" />
            <span className="text-xs text-slate-500">mm</span>
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400 shrink-0">Sensor H</span>
            <input type="number" min={1} max={200} step={0.01}
              disabled={running}
              value={cameraSettings.sensorHeightMm}
              onChange={e => onCameraSettingsChange({ ...cameraSettings, sensorHeightMm: e.target.value })}
              placeholder="24.0"
              className="w-20 px-2 py-1 rounded-md border border-slate-600 text-sm text-slate-200 bg-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50" />
            <span className="text-xs text-slate-500">mm</span>
          </label>
        </div>
      </div>

      {/* ── FL setup bar ─────────────────────────────────────────────── */}
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-4 space-y-3">

        {/* Chart setup mode */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider shrink-0">Chart Setup</span>
          {([false, true] as const).map(fm => (
            <button key={String(fm)} type="button" disabled={running}
              onClick={() => setFixedMount(fm)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors border ${
                fixedMount === fm
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600'
              }`}>
              {fm ? 'Fixed Chart' : 'Move Chart'}
            </button>
          ))}
          {fixedMount && (
            <span className="text-[11px] text-amber-400/80 border border-amber-500/30 bg-amber-500/10 rounded px-2 py-0.5">
              Tripod only — pan/tilt/zoom, never walk closer
            </span>
          )}
        </div>

        {/* Zoom mode selector */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider shrink-0">Zoom Mode</span>
          {(['2pt', '3pt', 'manual'] as const).map(m => (
            <button key={m} type="button" disabled={running}
              onClick={() => setZoomMode(m)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors border ${
                zoomMode === m
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600'
              }`}>
              {m === '2pt' ? 'Min → Max' : m === '3pt' ? 'Min + Mid + Max' : 'Manual'}
            </button>
          ))}
        </div>

        {/* Guided mode: enter zoom range, auto-fill FLs */}
        {(zoomMode === '2pt' || zoomMode === '3pt') && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <label className="text-[11px] text-slate-400 shrink-0">Wide end (mm)</label>
              <input type="number" min={1} placeholder="e.g. 28" value={zoomMinInput}
                onChange={e => setZoomMinInput(e.target.value)}
                disabled={running}
                className="w-24 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500" />
              {zoomMode === '3pt' && (
                <>
                  <label className="text-[11px] text-slate-400 shrink-0">Mid (mm)</label>
                  <input type="number" min={1} placeholder="e.g. 50" value={zoomMidInput}
                    onChange={e => setZoomMidInput(e.target.value)}
                    disabled={running}
                    className="w-24 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500" />
                </>
              )}
              <label className="text-[11px] text-slate-400 shrink-0">Tele end (mm)</label>
              <input type="number" min={1} placeholder="e.g. 100" value={zoomMaxInput}
                onChange={e => setZoomMaxInput(e.target.value)}
                disabled={running}
                className="w-24 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500" />
              <button type="button" disabled={running}
                onClick={() => {
                  const mn = parseFloat(zoomMinInput);
                  const mx = parseFloat(zoomMaxInput);
                  if (!mn || !mx || mn >= mx) return;
                  let pts: number[];
                  if (zoomMode === '2pt') {
                    pts = [mn, mx];
                  } else {
                    const mid = parseFloat(zoomMidInput);
                    const midVal = (mid && mid > mn && mid < mx) ? mid : Math.round((mn + mx) / 2);
                    pts = [mn, midVal, mx];
                  }
                  const newGroups = pts.map(fl => ({ fl_mm: fl, frames: [] as ScoredFrame[], status: 'pending' as const, working_distance_mm: 0 }));
                  setFlGroups(newGroups);
                  setActiveFlIdx(0);
                }}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white rounded text-xs font-semibold transition-colors">
                Set focal lengths
              </button>
            </div>
            <p className="text-[11px] text-slate-500">
              {zoomMode === '2pt'
                ? 'Captures at wide + tele only. Physics-informed interpolation fills every mm between them — fastest workflow.'
                : 'Adds a mid-range point for a tighter fit. Recommended for lenses with strong distortion variation across the zoom range.'}
            </p>
          </div>
        )}

        {/* Manual mode: snap buttons + custom input */}
        {zoomMode === 'manual' && (
          <div className="space-y-2">
            <div className="flex items-center gap-3 flex-wrap">
              <div className="flex gap-1 flex-wrap">
                {SNAP_FLS.map(fl => (
                  <button key={fl} type="button" onClick={() => addFl(fl)}
                    disabled={running || flGroups.some(g => g.fl_mm === fl)}
                    className="px-2 py-0.5 rounded text-[11px] bg-slate-700 hover:bg-slate-600 disabled:opacity-40 text-slate-300 transition-colors">
                    {fl}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1 ml-auto">
                <input type="number" min={1} placeholder="custom mm" value={flInput}
                  onChange={e => setFlInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') { addFl(parseFloat(flInput)); setFlInput(''); } }}
                  className="w-24 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500" />
                <button type="button" onClick={() => { addFl(parseFloat(flInput)); setFlInput(''); }} disabled={running}
                  className="px-2 py-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white rounded text-xs transition-colors">+</button>
              </div>
            </div>
          </div>
        )}

        <p className="text-[11px] text-slate-500">
          {fixedMount
            ? <>Fixed chart on a stand — mount camera on a tripod and only pan, tilt, or zoom. Set working distance so the chart is <span className="text-slate-300">visible at your widest focal length</span>, then capture all FLs without moving the camera body.</>
            : flGroups.length > 1
              ? <><span className="text-slate-400 font-medium">Zoom lens tip:</span> at longer focal lengths the chart will overflow the frame. Step back so it fills ~60% of frame height at each FL — distance scales with the zoom ratio. Aim for {MIN_FRAMES_PER_FL}+ frames each.</>
              : <>Keep the camera at the same working distance for all focal lengths — only rotate the zoom ring. Aim for {MIN_FRAMES_PER_FL}+ frames each.</>
          }
        </p>
      </div>

      {/* ── Main area: video + checklist/FL-list ─────────────────────── */}
      <div className="flex gap-4 items-start flex-wrap lg:flex-nowrap">

        {/* Left: video */}
        <div className="flex-shrink-0 space-y-3 w-full lg:w-[640px]">
          <div className="rounded-xl overflow-hidden bg-slate-900 border border-slate-700 relative aspect-video">
            {liveFrame ? (
              <img src={`data:image/jpeg;base64,${liveFrame.frame}`} alt="Live" className="absolute inset-0 w-full h-full object-fill" draggable={false} />
            ) : previewFrame ? (
              <img src={`data:image/jpeg;base64,${previewFrame}`} alt="Preview" className="absolute inset-0 w-full h-full object-fill" draggable={false} />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-sm">
                {previewing || running ? 'Connecting to camera…' : selectedIdx !== null ? 'Loading preview…' : 'No camera selected'}
              </div>
            )}
            <canvas ref={canvasRef} width={STREAM_W} height={STREAM_H} className="absolute inset-0 w-full h-full pointer-events-none" />
            <div className={`absolute inset-0 bg-white pointer-events-none transition-opacity duration-500 ${flash ? 'opacity-60' : 'opacity-0'}`} />
            {liveFrame && liveFrame.hold_progress > 0 && !liveFrame.complete && <HoldRing progress={liveFrame.hold_progress} />}
            {liveFrame && (
              <div className={`absolute top-2 right-2 text-[10px] font-bold px-2 py-1 rounded-lg border capitalize ${liveFrame.found ? (QUALITY_COLOR[liveFrame.quality] ?? '') : 'text-slate-500 border-slate-600/40 bg-black/40'}`}>
                {liveFrame.found ? liveFrame.quality : 'no chart'}
              </div>
            )}
            {running && <div className="absolute top-2 left-2 text-[10px] font-bold px-2 py-1 rounded-lg bg-black/60 text-white">{frameCount} captured{currentFl ? ` · ${currentFl}mm` : ''}</div>}
            {!running && previewFrame && <div className="absolute top-2 left-2 text-[10px] font-bold px-2 py-1 rounded-lg bg-black/50 text-slate-400 tracking-wider">PREVIEW</div>}
            {checklistComplete && running && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="bg-emerald-600 text-white rounded-xl px-6 py-4 text-center shadow-2xl">
                  <div className="text-2xl mb-1">All poses captured!</div>
                  <div className="text-sm opacity-90">Press Stop to move to next focal length</div>
                </div>
              </div>
            )}
          </div>

          {/* Progress bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{satisfiedCount} / {totalPoses} poses</span>
              <span className="text-slate-500">{frameCount} frames</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <progress value={progressPct} max={100}
                className={`block w-full h-full [appearance:none] [&::-webkit-progress-bar]:bg-transparent [&::-webkit-progress-value]:rounded-full [&::-webkit-progress-value]:transition-all ${checklistComplete ? '[&::-webkit-progress-value]:bg-emerald-500 [&::-moz-progress-bar]:bg-emerald-500' : '[&::-webkit-progress-value]:bg-blue-500  [&::-moz-progress-bar]:bg-blue-500'}`} />
            </div>
          </div>

          {/* Next hint + chips */}
          {running && nextHint && (
            <div className={`rounded-lg border px-3 py-3 flex items-center gap-4 ${checklistComplete ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-slate-800 border-slate-700'}`}>
              {!checklistComplete && liveFrame?.next_pose_id && <PoseDiagram poseId={liveFrame.next_pose_id} size="lg" />}
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-semibold ${checklistComplete ? 'text-emerald-300' : 'text-slate-200'}`}>
                  {checklistComplete ? '✓ All poses captured!' : nextHint}
                </p>
                {!checklistComplete && liveFrame?.match_status && (
                  <div className="flex gap-1.5 mt-2 flex-wrap">
                    {(['position_ok', 'tilt_ok', 'size_ok'] as const).map(key => {
                      const ok = liveFrame.match_status![key];
                      const label = key === 'position_ok' ? 'Position' : key === 'tilt_ok' ? 'Tilt' : 'Size';
                      return (
                        <span key={key} className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${ok ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300' : 'bg-slate-700/60 border-slate-600 text-slate-400'}`}>
                          {ok ? '✓' : '·'} {label}
                        </span>
                      );
                    })}
                  </div>
                )}
                {!checklistComplete && liveFrame?.match_status?.position_hint && (
                  <p className="text-sm font-bold text-yellow-300 mt-2">
                    {liveFrame.match_status.position_hint.includes('up') && !liveFrame.match_status.position_hint.includes('down') ? '↑ ' : ''}
                    {liveFrame.match_status.position_hint.includes('down') && !liveFrame.match_status.position_hint.includes('up') ? '↓ ' : ''}
                    {liveFrame.match_status.position_hint.includes('left') && !liveFrame.match_status.position_hint.includes('right') ? '← ' : ''}
                    {liveFrame.match_status.position_hint.includes('right') && !liveFrame.match_status.position_hint.includes('left') ? '→ ' : ''}
                    {liveFrame.match_status.position_hint}
                  </p>
                )}
                {!checklistComplete && liveFrame?.match_status?.tilt_hint && liveFrame.match_status.tilt_hint !== 'good' && (
                  <p className="text-xs text-slate-400 mt-1.5">
                    Tilt: <span className="text-slate-200 font-semibold">{liveFrame.match_status.tilt_hint}</span>
                    {typeof liveFrame.match_status.tilt_score === 'number' && (
                      <span className="text-slate-500"> · {liveFrame.match_status.tilt_score.toFixed(2)}</span>
                    )}
                  </p>
                )}
                {!checklistComplete && liveFrame?.match_status?.size_hint && liveFrame.match_status.size_hint !== 'good' && (
                  <p className="text-xs text-slate-500 mt-1">
                    Distance guide: <span className="text-slate-300 font-semibold">{liveFrame.match_status.size_hint}</span>
                    {typeof liveFrame.match_status.size_score === 'number' && (
                      <span className="text-slate-500"> · size {liveFrame.match_status.size_score.toFixed(3)}</span>
                    )}
                  </p>
                )}
                {!checklistComplete && liveFrame && !liveFrame.found && <p className="text-xs text-slate-500 mt-1.5">Point the chart at the camera</p>}
                {checklistComplete && <p className="text-xs text-slate-400 mt-0.5">Press Stop to move to next focal length</p>}
              </div>
            </div>
          )}

          {/* Post-FL-capture hint */}
          {!running && activeFlIdx !== null && flGroups[activeFlIdx]?.status === 'done' && nextPendingFlIdx >= 0 && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="text-emerald-400">✓ {flGroups[activeFlIdx].frames.filter(f => f.quality !== 'fail').length} frames saved</span>
              <span>·</span>
              <span>Select <span className="text-blue-300 font-semibold">{flGroups[nextPendingFlIdx].fl_mm}mm</span> in the list to continue</span>
            </div>
          )}

          {handoffStatus && !running && (
            <div className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-300">
              {handoffStatus}
            </div>
          )}

          {/* Single-FL: Run Calibration button */}
          {singleFlReady && !running && (
            <div className="flex items-center gap-3">
              <button type="button" onClick={runCalibration} disabled={!canCalibrateSingle || calibrating}
                className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors ${canCalibrateSingle && !calibrating ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-slate-700 text-slate-500 cursor-not-allowed'}`}>
                {calibrating
                  ? 'Calibrating…'
                  : `Run Calibration (${flGroups[activeFlIdx!].frames.filter(f => f.quality !== 'fail').length} frames)`}
              </button>
              {calibrating && <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />}
              {checklistComplete && !calibrating && <span className="text-xs text-emerald-400">All 10 poses captured — ready!</span>}
            </div>
          )}
        </div>

        {/* Right: pose checklist + FL step list */}
        <div className="flex-1 min-w-[220px] space-y-3">

          {/* Pose checklist — 5-per-row grid, 2 rows for 10 poses */}
          {(running ? (liveFrame?.checklist ?? checklist) : checklist).length > 0 && (
            <div className="rounded-xl bg-slate-800 border border-slate-700 p-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Required Poses</h3>
              {running && !checklistComplete && <p className="text-[10px] text-slate-500 mb-3">Click an unsatisfied pose to target it manually</p>}
              <div className="grid grid-cols-5 gap-2">
                {(running ? (liveFrame?.checklist ?? checklist) : checklist).map(pose => {
                  const isPinned = pinnedPoseId === pose.id;
                  return (
                  <div key={pose.id}
                    title={pose.satisfied ? pose.name : `${pose.name}: ${pose.hint}`}
                    onClick={() => running && !pose.satisfied && selectPose(pose.id)}
                    className={`flex flex-col items-center gap-1 p-2 rounded-lg border transition-colors ${
                      pose.satisfied
                        ? 'bg-emerald-500/10 border-emerald-500/20'
                        : isPinned
                          ? 'bg-blue-500/20 border-blue-400/60 cursor-pointer ring-1 ring-blue-400/60'
                          : running
                            ? 'bg-slate-700/50 border-transparent cursor-pointer hover:bg-slate-600/60 hover:border-slate-500'
                            : 'bg-slate-700/50 border-transparent'
                    }`}>
                    <PoseDiagram poseId={pose.id} size="sm" satisfied={pose.satisfied} />
                    <p className={`text-[9px] font-semibold text-center leading-tight line-clamp-2 ${
                      pose.satisfied ? 'text-emerald-300' : isPinned ? 'text-blue-300' : 'text-slate-400'
                    }`}>{pose.name}</p>
                    {pose.satisfied
                      ? <span className="text-[9px] text-emerald-400">✓</span>
                      : isPinned
                        ? <span className="text-[9px] text-blue-400 font-bold">●</span>
                        : null}
                  </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* FL step list */}
          <div className="rounded-xl bg-slate-800 border border-slate-700 p-4 space-y-3">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Focal Length Steps</h3>

            {flGroups.length === 0 ? (
              <p className="text-xs text-slate-500 italic">Add focal lengths above</p>
            ) : (
              <div className="space-y-2">
                {(() => {
                  const wideFl = flGroups.length > 0 ? Math.min(...flGroups.map(g => g.fl_mm)) : null;
                  return flGroups.map((g, i) => {
                  const goodFrames  = g.frames.filter(f => f.quality !== 'fail').length;
                  const capturedCount = g.frames.length;
                  const ready       = g.status === 'done' && goodFrames >= MIN_FRAMES_PER_FL;
                  const isActive    = activeFlIdx === i;
                  const isCapturing = isActive && running;
                  // Step-back ratio relative to the widest FL (only shown for zoom setups)
                  const ratio = wideFl && flGroups.length > 1 && g.fl_mm > wideFl
                    ? (g.fl_mm / wideFl) : null;
                  return (
                    <div key={g.fl_mm} onClick={() => !running && setActiveFlIdx(i)}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors ${
                        isCapturing ? 'bg-blue-500/15 border-blue-500/40' :
                        isActive    ? 'bg-slate-700/80 border-slate-600' :
                        ready       ? 'bg-emerald-500/10 border-emerald-500/20' :
                                      'bg-slate-700/50 border-transparent hover:bg-slate-700'
                      }`}>
                      <span className={`text-base leading-none ${ready ? 'text-emerald-400' : isCapturing ? 'text-blue-400' : isActive ? 'text-slate-400' : 'text-slate-600'}`}>
                        {ready ? '✓' : isCapturing ? '●' : '○'}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-baseline gap-2 flex-wrap">
                          <span className={`text-sm font-semibold ${ready ? 'text-emerald-300' : isActive ? 'text-blue-300' : 'text-slate-300'}`}>
                            {g.fl_mm} mm
                          </span>
                          {ratio !== null && (
                            <span
                              className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 font-medium"
                              title={`Step back to ≈${ratio.toFixed(1)}× your ${wideFl}mm distance so the chart fills ~60% of frame`}>
                              {ratio.toFixed(1)}× distance
                            </span>
                          )}
                        </div>
                        {g.status === 'done' && (
                          <span className="text-[10px] text-slate-500">
                            {capturedCount} captured · {goodFrames} usable
                          </span>
                        )}
                        {g.status === 'pending' && !isActive && <span className="text-[10px] text-slate-600">pending</span>}
                        {isActive && !running && g.status !== 'done' && <span className="text-[10px] text-blue-400">ready to capture</span>}
                        {isCapturing && <span className="text-[10px] text-blue-400">capturing…</span>}
                        <div className="mt-1 flex items-center gap-1.5">
                          <label className="text-[10px] text-slate-500" htmlFor={`wd-${g.fl_mm}`}>WD</label>
                          <input
                            id={`wd-${g.fl_mm}`}
                            type="number"
                            min={0}
                            step="10"
                            inputMode="decimal"
                            value={g.working_distance_mm > 0 ? String(g.working_distance_mm) : ''}
                            placeholder="mm"
                            disabled={running}
                            onClick={e => e.stopPropagation()}
                            onChange={e => setFlWorkingDistance(g.fl_mm, e.target.value)}
                            className="w-20 rounded border border-slate-600 bg-slate-900 px-1.5 py-0.5 text-[10px] text-slate-200 placeholder:text-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-60"
                            title="Camera-to-chart working distance in mm for this focal-length group"
                          />
                          <span className="text-[10px] text-slate-600">mm</span>
                        </div>
                      </div>
                      {!running && (
                        <button type="button" onClick={e => { e.stopPropagation(); removeFl(g.fl_mm); }}
                          className="text-slate-600 hover:text-red-400 text-sm transition-colors" title="Remove">×</button>
                      )}
                    </div>
                  );
                });
                })()}
              </div>
            )}

            {/* Capture controls for the selected FL */}
            {activeFlIdx !== null && flGroups[activeFlIdx] && (
              <div className="pt-3 border-t border-slate-700 space-y-2">
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                  {flGroups[activeFlIdx].fl_mm}mm
                </p>
                {!running ? (
                  <button type="button" onClick={startCapture}
                    disabled={!ws || selectedIdx === null}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg text-sm font-medium transition-colors">
                    {flGroups[activeFlIdx].status === 'done'
                      ? `Re-capture ${flGroups[activeFlIdx].fl_mm}mm`
                      : `Start Capture at ${flGroups[activeFlIdx].fl_mm}mm`}
                  </button>
                ) : (
                  <div className="flex gap-2">
                    <button type="button" onClick={manualCapture}
                      className="flex-1 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors">Capture</button>
                    <button type="button" onClick={stopCapture}
                      className="flex-1 px-3 py-2 bg-red-600/80 hover:bg-red-600 text-white rounded-lg text-sm font-medium transition-colors">Stop</button>
                  </div>
                )}

                {activeFlDiversity && (
                  <div className={`rounded-lg border px-2.5 py-2 text-[10px] ${
                    activeFlDiversity.level === 'good'
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                      : activeFlDiversity.level === 'fair'
                        ? 'border-yellow-500/40 bg-yellow-500/10 text-yellow-300'
                        : 'border-red-500/40 bg-red-500/10 text-red-300'
                  }`}>
                    <div className="flex items-center justify-between">
                      <span className="font-semibold uppercase tracking-wider">Capture Diversity</span>
                      <span className="font-mono tabular-nums">{activeFlDiversity.score}/100</span>
                    </div>
                    <div className="mt-1 grid grid-cols-3 gap-2 text-[9px] text-slate-300">
                      <span>Center Δ {activeFlDiversity.centerSpread.toFixed(2)}</span>
                      <span>Tilt Δ {activeFlDiversity.tiltSpreadDeg.toFixed(1)}°</span>
                      <span>Scale Δ {activeFlDiversity.coverageSpread.toFixed(3)}</span>
                    </div>
                    {activeFlDiversity.level !== 'good' && (
                      <p className="mt-1 text-[9px] text-slate-300">
                        Add more varied frames: left/right + high/low + stronger tilt at this FL.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Zoom calibration button (≥2 FLs done) */}
            {flGroups.length >= 2 && (
              <div className="pt-3 border-t border-slate-700">
                {readyFlCount >= 2 ? (
                  <button type="button" onClick={runZoomCalibration} disabled={!canCalibrateZ}
                    className={`w-full px-4 py-2.5 rounded-lg text-sm font-semibold transition-colors ${canCalibrateZ ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-slate-700 text-slate-500 cursor-not-allowed'}`}>
                    {calibratingZoom ? 'Calibrating…' : `Run Zoom Calibration (${readyFlCount} FLs)`}
                  </button>
                ) : (
                  <p className="text-xs text-slate-500 text-center">
                    Capture {Math.max(0, 2 - readyFlCount)} more FL{readyFlCount < 1 ? 's' : ''} to enable calibration
                  </p>
                )}
                {calibratingZoom && <div className="flex justify-center mt-2"><div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" /></div>}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Zoom calibration results ──────────────────────────────────── */}
      {zoomResult && (
        <div className="rounded-xl bg-slate-800 border border-slate-700 p-5 space-y-5">
          <h2 className="text-sm font-semibold text-slate-300 tracking-wide uppercase">Zoom Calibration Results</h2>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-700">
                  <th className="text-left pb-2 pr-4">FL (mm)</th>
                  <th className="text-right pb-2 pr-4">RMS</th>
                  <th className="text-right pb-2 pr-4">f (px)</th>
                  <th className="text-right pb-2 pr-4">f (mm)</th>
                  <th className="text-right pb-2 pr-4">k1</th>
                  <th className="text-right pb-2 pr-4">k2</th>
                  <th className="text-right pb-2 pr-4">Nodal &#916;</th>
                  <th className="text-right pb-2">Quality</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {zoomResult.fl_results.map(r => {
                  const nz = zoomResult.nodal_offsets_mm[String(r.focal_length_mm)];
                  return (
                    <tr key={r.focal_length_mm} className={r.error ? 'opacity-50' : ''}>
                      <td className="py-2 pr-4 font-semibold text-slate-200">{r.focal_length_mm}</td>
                      <td className={`py-2 pr-4 text-right font-mono tabular-nums ${r.rms == null ? 'text-red-400' : r.rms < 0.5 ? 'text-emerald-400' : r.rms < 1.0 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {r.rms != null ? r.rms.toFixed(3) : '—'}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-300">{r.fx_px ? r.fx_px.toFixed(0) : '—'}</td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-400">{r.focal_length_computed_mm ? r.focal_length_computed_mm.toFixed(1) : '—'}</td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-300">{r.dist_coeffs?.[0] != null ? r.dist_coeffs[0].toFixed(4) : '—'}</td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-300">{r.dist_coeffs?.[1] != null ? r.dist_coeffs[1].toFixed(4) : '—'}</td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums text-slate-400">{nz != null ? `${nz >= 0 ? '+' : ''}${nz.toFixed(1)} mm` : '—'}</td>
                      <td className={`py-2 text-right capitalize ${CONFIDENCE_COLOR[r.confidence] ?? 'text-slate-500'}`}>{r.error ? 'error' : (r.confidence ?? '—')}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-slate-500 leading-relaxed">
            Nodal &#916; = estimated optical-centre shift along Z-axis relative to the shortest focal length.
          </p>

          {/* Export */}
          <div className="pt-3 border-t border-slate-700 space-y-3">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Export UE5 .ulens (Zoom)</h3>
            {(!cameraSettings.sensorWidthMm || !cameraSettings.lensName) && (
              <p className="text-[11px] text-yellow-500/80">Set sensor dimensions and lens name in Settings for accurate export.</p>
            )}
            <button type="button" onClick={doZoomExport} disabled={exportStatus === 'loading' || !ws}
              className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors ${
                exportStatus === 'success' ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' :
                exportStatus === 'error'   ? 'bg-red-500/10 border border-red-500/30 text-red-400' :
                exportStatus === 'loading' ? 'bg-slate-700 text-slate-400 cursor-wait border border-slate-600' :
                                             'bg-blue-600 hover:bg-blue-500 text-white'
              }`}>
              {exportStatus === 'loading' ? 'Exporting…' : exportStatus === 'success' ? 'Exported' : exportStatus === 'error' ? 'Export failed' : 'Export UE5 .ulens (multi-FL)'}
            </button>
            {exportPath && <p className="text-[10px] text-slate-500 font-mono truncate" title={exportPath}>{exportPath}</p>}
          </div>
        </div>
      )}
    </div>
  );
}
