import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, CameraSettings, LensSettings, ScoredFrame } from '../types';
import type { ZoomCalibResult } from './GuidedCapture';
import BoardPresetSelector from './BoardPresetSelector';
import FrameGrid from './FrameGrid';

const SENSOR_PRESETS = [
  // CineAlta
  { label: 'Venice — FF 6K (36.0 × 24.0)',           w: 36.0,  h: 24.0  },
  { label: 'Venice 2 / Burano — FF 8K (35.9 × 24.0)',w: 35.9,  h: 24.0  },
  { label: 'Venice 2 / Burano — S35 (26.2 × 14.7)', w: 26.2,  h: 14.7  },
  { label: 'Venice 2 / Burano — S16 (14.6 × 8.2)',  w: 14.6,  h: 8.2   },
  // Broadcast B4
  { label: 'HDC-series 2/3" HD B4 (9.59 × 5.39)',   w: 9.59,  h: 5.39  },
  { label: 'HDC-F5500 / HDC-3500 4K 2/3" (9.59 × 5.39)', w: 9.59, h: 5.39 },
  { label: 'HDC-5500 / HDW-series 2/3" (9.59 × 5.39)',   w: 9.59, h: 5.39 },
];

// ── constants ─────────────────────────────────────────────────────────────────
const ACCEPTED_EXT = new Set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']);
const MIN_FRAMES_PER_GROUP = 5;

function hasCalibExt(name: string) {
  const i = name.lastIndexOf('.');
  return ACCEPTED_EXT.has(i >= 0 ? name.slice(i).toLowerCase() : '');
}

const electronAPI = (window as unknown as {
  electronAPI?: {
    showOpenDialog: (opts: object) => Promise<{ canceled: boolean; filePaths: string[] }>;
  };
}).electronAPI;

// ── types ─────────────────────────────────────────────────────────────────────
interface FlGroup {
  id: number;
  fl_mm: number;
  frames: ScoredFrame[];
  excluded: Set<string>;
}

type PreviewFrame = {
  path: string; quality: 'good' | 'warn' | 'fail';
  sharpness: number; coverage: number; index: number;
};

export interface Props {
  ws: WebSocket | null;
  boardSettings: BoardSettings;
  onBoardChange: (s: BoardSettings) => void;
  lensSettings: LensSettings;
  cameraSettings: CameraSettings;
  onCameraSettingsChange: (s: CameraSettings) => void;
  backendPort: number;
  calibrating: boolean;
  setCalibrating: (v: boolean) => void;
  onZoomCalibrationComplete: (result: ZoomCalibResult, imageSize: [number, number]) => void;
  onSingleCalibrationSent: (imageSize: [number, number], frames: PreviewFrame[]) => void;
}

// ── stable group factory ───────────────────────────────────────────────────────
let _gid = 1;
const mkGroup = (fl_mm: number): FlGroup => ({ id: _gid++, fl_mm, frames: [], excluded: new Set() });

// ── component ─────────────────────────────────────────────────────────────────
export default function FileCalibration({
  ws, boardSettings, onBoardChange, lensSettings, cameraSettings, onCameraSettingsChange,
  backendPort, calibrating, setCalibrating, onZoomCalibrationComplete, onSingleCalibrationSent,
}: Props) {
  const [groups, setGroups] = useState<FlGroup[]>(() => [mkGroup(28), mkGroup(100)]);
  const [scoring, setScoring] = useState(false);
  const [scoringProgress, setScoringProgress] = useState({ done: 0, total: 0 });
  const [calibratingZoom, setCalibratingZoom] = useState(false);
  const [draggingGroupId, setDraggingGroupId] = useState<number | null>(null);

  // Refs for mid-flight scoring state — stable across renders
  const scoringGroupIdRef = useRef<number | null>(null);
  const scoringPathsRef   = useRef<Set<string>>(new Set());
  const pendingFramesRef  = useRef<ScoredFrame[]>([]);
  const groupsRef         = useRef(groups);
  useEffect(() => { groupsRef.current = groups; }, [groups]);

  // ── WS listener ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);

        if (msg.action === 'frame_result') {
          // Only handle frames we submitted from this component
          if (scoringGroupIdRef.current === null) return;
          if (!scoringPathsRef.current.has(msg.path)) return;
          pendingFramesRef.current.push(msg as ScoredFrame);
          setScoringProgress(p => ({ ...p, done: p.done + 1 }));

        } else if (msg.action === 'score_frames_done') {
          const gid = scoringGroupIdRef.current;
          if (gid === null) return;                  // not our session
          const scored = [...pendingFramesRef.current];
          pendingFramesRef.current = [];
          scoringGroupIdRef.current = null;
          scoringPathsRef.current = new Set();
          setScoring(false);
          // Merge into group, deduplicating on path
          setGroups(prev => prev.map(g => {
            if (g.id !== gid) return g;
            const existing = new Set(g.frames.map(f => f.path));
            const novel = scored.filter(f => !existing.has(f.path));
            return { ...g, frames: [...g.frames, ...novel] };
          }));

        } else if (msg.action === 'zoom_calibrate_result') {
          setCalibratingZoom(false);
          const allFrames = groupsRef.current.flatMap(g => g.frames);
          const img: [number, number] = allFrames[0]
            ? [allFrames[0].image_width, allFrames[0].image_height]
            : [1920, 1080];
          onZoomCalibrationComplete(msg as ZoomCalibResult, img);
        }
      } catch { /* ignore */ }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws, onZoomCalibrationComplete]);

  // ── scoring helpers ──────────────────────────────────────────────────────────
  const scoreFiles = (groupId: number, paths: string[]) => {
    if (!ws || ws.readyState !== WebSocket.OPEN || scoring || paths.length === 0) return;
    scoringGroupIdRef.current = groupId;
    scoringPathsRef.current = new Set(paths);
    pendingFramesRef.current = [];
    setScoring(true);
    setScoringProgress({ done: 0, total: paths.length });
    ws.send(JSON.stringify({
      action: 'score_frames',
      paths,
      board_cols: boardSettings.cols,
      board_rows: boardSettings.rows,
    }));
  };

  const browseFiles = async (groupId: number) => {
    if (scoring || !electronAPI) return;
    const res = await electronAPI.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Calibration Images', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] }],
    });
    if (!res.canceled && res.filePaths.length > 0) {
      scoreFiles(groupId, res.filePaths.filter(hasCalibExt));
    }
  };

  const browseFolder = async (groupId: number) => {
    if (scoring || !electronAPI) return;
    const res = await electronAPI.showOpenDialog({ properties: ['openDirectory'] });
    if (res.canceled || res.filePaths.length === 0) return;
    const second = await electronAPI.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      defaultPath: res.filePaths[0],
      filters: [{ name: 'Calibration Images', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] }],
    });
    if (!second.canceled && second.filePaths.length > 0) {
      scoreFiles(groupId, second.filePaths.filter(hasCalibExt));
    }
  };

  const handleDrop = (e: React.DragEvent, groupId: number) => {
    e.preventDefault();
    setDraggingGroupId(null);
    if (scoring) return;
    const paths = Array.from(e.dataTransfer.files)
      .filter(f => hasCalibExt(f.name))
      .map(f => (f as unknown as { path?: string }).path ?? '')
      .filter(Boolean);
    scoreFiles(groupId, paths);
  };

  // ── group management ─────────────────────────────────────────────────────────
  const addGroup    = () => setGroups(prev => [...prev, mkGroup(50)]);
  const removeGroup = (id: number) =>
    setGroups(prev => prev.length > 1 ? prev.filter(g => g.id !== id) : prev);
  const clearGroup  = (id: number) =>
    setGroups(prev => prev.map(g => g.id === id ? { ...g, frames: [], excluded: new Set() } : g));
  const toggleExclude = (groupId: number, path: string) =>
    setGroups(prev => prev.map(g => {
      if (g.id !== groupId) return g;
      const next = new Set(g.excluded);
      next.has(path) ? next.delete(path) : next.add(path);
      return { ...g, excluded: next };
    }));

  // ── calibration ──────────────────────────────────────────────────────────────
  const isMultiFL  = groups.length >= 2;
  const readyGroups = groups.filter(g =>
    g.frames.filter(f => !g.excluded.has(f.path) && f.quality !== 'fail').length >= MIN_FRAMES_PER_GROUP
  );
  const canCalibrate = isMultiFL ? readyGroups.length >= 2 : readyGroups.length >= 1;
  const busy = calibrating || calibratingZoom || scoring;
  const anyFrames = groups.some(g => g.frames.length > 0);
  const sensorOk = parseFloat(cameraSettings.sensorWidthMm) > 0;
  const canFire = canCalibrate && (!isMultiFL || sensorOk);

  const runCalibration = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN || busy || !canFire) return;

    if (isMultiFL) {
      const allFrames = readyGroups.flatMap(g => g.frames);
      const imgSize: [number, number] = allFrames[0]
        ? [allFrames[0].image_width, allFrames[0].image_height]
        : [1920, 1080];
      setCalibratingZoom(true);
      ws.send(JSON.stringify({
        action: 'calibrate_zoom',
        fl_groups: readyGroups.map(g => ({
          focal_length_mm: g.fl_mm,
          frames: g.frames.filter(f => !g.excluded.has(f.path)),
        })),
        board_cols: boardSettings.cols,
        board_rows: boardSettings.rows,
        square_size_mm: boardSettings.squareSizeMm,
        image_size: imgSize,
        sensor_width_mm: parseFloat(cameraSettings.sensorWidthMm) || 0,
        sensor_height_mm: parseFloat(cameraSettings.sensorHeightMm) || 0,
        lens_type: lensSettings.lensType,
        squeeze_ratio: lensSettings.squeezeRatio,
      }));
    } else {
      const g = readyGroups[0];
      const includedFrames = g.frames.filter(f => !g.excluded.has(f.path));
      const imgSize: [number, number] = includedFrames[0]
        ? [includedFrames[0].image_width, includedFrames[0].image_height]
        : [1920, 1080];
      onSingleCalibrationSent(imgSize, includedFrames.map((f, i) => ({
        path: f.path, quality: f.quality, sharpness: f.sharpness, coverage: f.coverage, index: i,
      })));
      setCalibrating(true);
      ws.send(JSON.stringify({
        action: 'calibrate',
        scored_frames: includedFrames,
        board_cols: boardSettings.cols,
        board_rows: boardSettings.rows,
        square_size_mm: boardSettings.squareSizeMm,
        image_size: imgSize,
        lens_type: lensSettings.lensType,
        squeeze_ratio: lensSettings.squeezeRatio,
      }));
    }
  };

  const buttonLabel = () => {
    if (calibratingZoom) return 'Running zoom calibration…';
    if (calibrating) return 'Calibrating…';
    if (scoring) return 'Scoring frames…';
    if (isMultiFL) {
      if (!canCalibrate) return `Need ≥${MIN_FRAMES_PER_GROUP} usable frames per group`;
      if (!sensorOk) return 'Select a sensor preset to continue';
      const total = readyGroups.reduce(
        (s, g) => s + g.frames.filter(f => !g.excluded.has(f.path) && f.quality !== 'fail').length, 0
      );
      return `Run Zoom Calibration (${readyGroups.length} groups · ${total} frames)`;
    }
    const g = groups[0];
    const n = g.frames.filter(f => !g.excluded.has(f.path) && f.quality !== 'fail').length;
    return canCalibrate ? `Run Calibration (${n} frames)` : `Need ≥${MIN_FRAMES_PER_GROUP} usable frames`;
  };

  // ── render ───────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-4">

      {/* ── Sensor / camera ── */}
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-4 space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-slate-300">Camera Sensor</h2>
          {isMultiFL && !sensorOk && (
            <span className="text-[11px] font-semibold text-yellow-400 bg-yellow-400/10 border border-yellow-400/30 rounded-full px-2 py-0.5">
              Required for zoom calibration
            </span>
          )}
        </div>

        {/* Preset row */}
        <div className="flex flex-wrap gap-2">
          {SENSOR_PRESETS.map(p => {
            const active =
              cameraSettings.sensorWidthMm === String(p.w) &&
              cameraSettings.sensorHeightMm === String(p.h);
            return (
              <button
                key={p.label}
                onClick={() => onCameraSettingsChange({ ...cameraSettings, sensorWidthMm: String(p.w), sensorHeightMm: String(p.h) })}
                disabled={busy}
                className={`text-xs px-2.5 py-1 rounded-lg border transition-colors ${
                  active
                    ? 'bg-blue-600/30 border-blue-500/60 text-blue-300'
                    : 'bg-slate-700 border-slate-600 text-slate-400 hover:text-slate-200 hover:border-slate-500'
                }`}
              >
                {p.label}
              </button>
            );
          })}
        </div>

        {/* Manual inputs row */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400 shrink-0">Sensor W</span>
            <input
              type="number" min={1} max={200} step={0.01}
              value={cameraSettings.sensorWidthMm}
              onChange={e => onCameraSettingsChange({ ...cameraSettings, sensorWidthMm: e.target.value })}
              placeholder="36.0"
              className={`w-20 px-2 py-1 rounded-md border text-sm text-slate-200 bg-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500 ${
                sensorOk ? 'border-slate-600' : 'border-yellow-500/60'
              }`}
            />
            <span className="text-xs text-slate-500">mm</span>
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400 shrink-0">Sensor H</span>
            <input
              type="number" min={1} max={200} step={0.01}
              value={cameraSettings.sensorHeightMm}
              onChange={e => onCameraSettingsChange({ ...cameraSettings, sensorHeightMm: e.target.value })}
              placeholder="24.0"
              className="w-20 px-2 py-1 rounded-md border border-slate-600 text-sm text-slate-200 bg-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <span className="text-xs text-slate-500">mm</span>
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400 shrink-0">Lens name</span>
            <input
              type="text"
              value={cameraSettings.lensName}
              onChange={e => onCameraSettingsChange({ ...cameraSettings, lensName: e.target.value })}
              placeholder="e.g. Premista28-100"
              className="w-40 px-2 py-1 rounded-md border border-slate-600 text-sm text-slate-200 bg-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </label>
        </div>
      </div>

      {/* Board settings */}
      <div className="rounded-xl bg-slate-800 border border-slate-700 p-4">
        <BoardPresetSelector boardSettings={boardSettings} onBoardChange={onBoardChange} disabled={busy} />
      </div>

      {/* FL group list */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300">
            {isMultiFL ? 'Focal Length Groups' : 'Frames'}
          </h2>
          <button
            onClick={addGroup}
            disabled={busy}
            className="text-xs px-3 py-1.5 bg-blue-600/80 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg border border-blue-500/40 transition-colors"
          >
            + Add FL group
          </button>
        </div>

        {groups.map((g, idx) => {
          const isActivelyScoringThisGroup = scoring && scoringGroupIdRef.current === g.id;
          const usable = g.frames.filter(f => !g.excluded.has(f.path) && f.quality !== 'fail').length;
          const total  = g.frames.length;

          return (
            <div
              key={g.id}
              className={`rounded-xl border p-4 space-y-3 transition-colors ${
                draggingGroupId === g.id
                  ? 'border-blue-500 bg-blue-500/5'
                  : 'border-slate-700 bg-slate-800'
              }`}
              onDragOver={e => { e.preventDefault(); if (!scoring) setDraggingGroupId(g.id); }}
              onDragLeave={() => setDraggingGroupId(prev => prev === g.id ? null : prev)}
              onDrop={e => handleDrop(e, g.id)}
            >
              {/* Group header row */}
              <div className="flex items-center gap-3 flex-wrap">
                {groups.length > 1 && (
                  <span className="text-xs font-medium text-slate-500">Group {idx + 1}</span>
                )}
                {isMultiFL && (
                  <label className="flex items-center gap-1.5">
                    <span className="text-xs text-slate-400">FL</span>
                    <input
                      type="number" min={1} max={2000} step={1}
                      value={g.fl_mm}
                      onChange={e => {
                        const v = parseFloat(e.target.value);
                        if (!isNaN(v) && v > 0)
                          setGroups(prev => prev.map(gg => gg.id === g.id ? { ...gg, fl_mm: v } : gg));
                      }}
                      disabled={busy}
                      className="w-20 px-2 py-1 rounded-md bg-slate-700 border border-slate-600 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    <span className="text-xs text-slate-400">mm</span>
                  </label>
                )}

                <div className="flex-1" />

                {total > 0 && (
                  <span className={`text-xs font-medium ${
                    usable >= MIN_FRAMES_PER_GROUP ? 'text-emerald-400' : 'text-yellow-400'
                  }`}>
                    {usable}/{total} usable
                  </span>
                )}
                {total > 0 && !busy && (
                  <button
                    onClick={() => clearGroup(g.id)}
                    className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                    title="Clear all frames from this group"
                  >
                    Clear
                  </button>
                )}
                {groups.length > 1 && (
                  <button
                    onClick={() => removeGroup(g.id)}
                    disabled={busy}
                    className="text-xs text-red-400/70 hover:text-red-400 disabled:opacity-40 transition-colors"
                    title="Remove this group"
                  >
                    ✕ Remove
                  </button>
                )}
              </div>

              {/* Scoring progress or drop/browse row */}
              {isActivelyScoringThisGroup ? (
                <div className="flex flex-col items-center gap-2 py-4">
                  <div className="w-8 h-8 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
                  <p className="text-xs text-slate-300">
                    Scoring {scoringProgress.done} / {scoringProgress.total}…
                  </p>
                  <div className="w-40 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all"
                      style={{
                        width: `${scoringProgress.total > 0
                          ? (scoringProgress.done / scoringProgress.total) * 100
                          : 0}%`,
                      }}
                    />
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2 rounded-lg border border-dashed border-slate-600 bg-slate-700/30 px-4 py-3">
                  <span className="text-sm text-slate-500 flex-1 truncate">
                    {total === 0
                      ? 'Drop images here or browse'
                      : `${total} frame${total !== 1 ? 's' : ''} loaded — drop more to add`}
                  </span>
                  <button
                    onClick={() => browseFiles(g.id)}
                    disabled={scoring}
                    className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-300 rounded-lg border border-slate-600 transition-colors shrink-0"
                  >
                    Browse files
                  </button>
                  <button
                    onClick={() => browseFolder(g.id)}
                    disabled={scoring}
                    className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-300 rounded-lg border border-slate-600 transition-colors shrink-0"
                  >
                    Browse folder
                  </button>
                </div>
              )}

              {/* Frame grid */}
              {g.frames.length > 0 && !isActivelyScoringThisGroup && (
                <FrameGrid
                  frames={g.frames}
                  excluded={g.excluded}
                  onToggle={path => toggleExclude(g.id, path)}
                  backendPort={backendPort}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Calibrate button */}
      {anyFrames && (
        <div className="flex items-center gap-3 pt-1">
          <button
            onClick={runCalibration}
            disabled={!canFire || busy}
            className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors ${
              canFire && !busy
                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
            }`}
          >
            {buttonLabel()}
          </button>
          {(calibratingZoom || calibrating) && (
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          )}
        </div>
      )}
    </div>
  );
}
