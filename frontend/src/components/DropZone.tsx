import { useEffect, useRef, useState } from 'react';
import type { BoardSettings, ScoredFrame } from '../types';
import BoardPresetSelector from './BoardPresetSelector';

const ACCEPTED_EXT = new Set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']);

function ext(name: string) {
  const i = name.lastIndexOf('.');
  return i >= 0 ? name.slice(i).toLowerCase() : '';
}

function isCalibImage(name: string) {
  return ACCEPTED_EXT.has(ext(name));
}

// Electron's contextBridge exposes electronAPI; fall back gracefully in browser dev mode.
const electronAPI = (window as unknown as { electronAPI?: {
  showOpenDialog: (opts: object) => Promise<{ canceled: boolean; filePaths: string[] }>;
} }).electronAPI;

interface Props {
  ws: WebSocket | null;
  boardSettings: BoardSettings;
  onBoardChange: (s: BoardSettings) => void;
  onScoringDone: (frames: ScoredFrame[]) => void;
}

export default function DropZone({ ws, boardSettings, onBoardChange, onScoringDone }: Props) {
  const [dragging, setDragging] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const pendingFrames = useRef<ScoredFrame[]>([]);

  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.action === 'frame_result') {
          pendingFrames.current.push(msg as ScoredFrame);
          setProgress((p) => ({ ...p, done: p.done + 1 }));
        } else if (msg.action === 'score_frames_done') {
          setScoring(false);
          onScoringDone([...pendingFrames.current]);
          pendingFrames.current = [];
        }
      } catch { /* ignore */ }
    };
    ws.addEventListener('message', handler);
    return () => ws.removeEventListener('message', handler);
  }, [ws]);

  const startScoring = (paths: string[]) => {
    if (!ws || ws.readyState !== WebSocket.OPEN || paths.length === 0) return;
    pendingFrames.current = [];
    setScoring(true);
    setProgress({ done: 0, total: paths.length });
    ws.send(JSON.stringify({
      action: 'score_frames',
      paths,
      board_cols: boardSettings.cols,
      board_rows: boardSettings.rows,
    }));
  };

  // Drag-and-drop: Electron populates a non-standard `path` property on File objects
  // even with contextIsolation, because drag source is the OS (not the renderer).
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (scoring) return;
    const paths = Array.from(e.dataTransfer.files)
      .filter(f => isCalibImage(f.name))
      .map(f => (f as unknown as { path?: string }).path ?? '')
      .filter(Boolean);
    startScoring(paths);
  };

  // "Browse files" — uses Electron native open dialog via IPC to get real fs paths.
  const browseFiles = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (scoring || !electronAPI) return;
    const result = await electronAPI.showOpenDialog({
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Calibration Images', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (!result.canceled && result.filePaths.length > 0) {
      startScoring(result.filePaths.filter(p => isCalibImage(p)));
    }
  };

  // "Browse folder" — opens a folder picker and passes all matching image paths.
  const browseFolder = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (scoring || !electronAPI) return;
    const result = await electronAPI.showOpenDialog({
      properties: ['openDirectory'],
    });
    if (!result.canceled && result.filePaths.length > 0) {
      // Ask the backend to enumerate the folder — send the folder path as a
      // single-element list; the backend score_frames handler reads each path.
      // Instead, resolve client-side by sending the folder path with a trailing
      // slash so main.py can glob it. Actually we need individual file paths, so
      // we pass the folder path to a separate action.
      // Simplest approach: send folder path to backend via score_frames with
      // a flag, but that requires backend changes. Instead: use the native dialog
      // again with 'openFile' + 'multiSelections' defaulting to that folder.
      const folderPath = result.filePaths[0];
      const second = await electronAPI.showOpenDialog({
        properties: ['openFile', 'multiSelections'],
        defaultPath: folderPath,
        filters: [
          { name: 'Calibration Images', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] },
        ],
      });
      if (!second.canceled && second.filePaths.length > 0) {
        startScoring(second.filePaths.filter(p => isCalibImage(p)));
      }
    }
  };

  const spinnerPct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;

  return (
    <div className="space-y-3">
      <BoardPresetSelector
        boardSettings={boardSettings}
        onBoardChange={onBoardChange}
        disabled={scoring}
      />

      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          relative rounded-xl border-2 border-dashed transition-colors
          flex flex-col items-center justify-center gap-3 p-8 cursor-pointer select-none
          ${dragging
            ? 'border-blue-500 bg-blue-500/10'
            : scoring
            ? 'border-slate-600 bg-slate-700/30'
            : 'border-slate-600 bg-slate-800 hover:border-slate-500 hover:bg-slate-700/40'}
        `}
        onClick={browseFiles}
      >
        {scoring ? (
          <>
            <div className="w-10 h-10 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
            <p className="text-sm text-slate-300">
              Scoring {progress.done} / {progress.total} frames…
            </p>
            <div className="w-48 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${spinnerPct}%` }}
              />
            </div>
          </>
        ) : (
          <>
            <div className="text-4xl text-slate-600">⊕</div>
            <div className="text-center">
              <p className="text-sm text-slate-300 font-medium">Drop calibration images here</p>
              <p className="text-xs text-slate-500 mt-1">JPG · PNG · DNG · TIFF</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={browseFiles}
                className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition-colors"
              >
                Browse files
              </button>
              <button
                onClick={browseFolder}
                className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition-colors"
              >
                Browse folder
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
