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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const pendingFrames = useRef<ScoredFrame[]>([]);
  const pendingTotal = useRef(0);

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
    pendingTotal.current = paths.length;
    setScoring(true);
    setProgress({ done: 0, total: paths.length });
    ws.send(JSON.stringify({
      action: 'score_frames',
      paths,
      board_cols: boardSettings.cols,
      board_rows: boardSettings.rows,
    }));
  };

  const extractPaths = (files: FileList | null): string[] => {
    if (!files) return [];
    return Array.from(files)
      .filter(f => isCalibImage(f.name))
      .map(f => (f as unknown as { path?: string }).path ?? '')
      .filter(Boolean);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (scoring) return;
    startScoring(extractPaths(e.dataTransfer.files));
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (scoring) return;
    startScoring(extractPaths(e.target.files));
    e.target.value = '';
  };

  const spinnerPct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;

  return (
    <div className="space-y-3">
      {/* Board preset */}
      <BoardPresetSelector
        boardSettings={boardSettings}
        onBoardChange={onBoardChange}
        disabled={scoring}
      />

      {/* Drop target */}
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
        onClick={() => !scoring && fileInputRef.current?.click()}
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
                onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}
                className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition-colors"
              >
                Browse files
              </button>
              <button
                onClick={e => { e.stopPropagation(); folderInputRef.current?.click(); }}
                className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition-colors"
              >
                Browse folder
              </button>
            </div>
          </>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".jpg,.jpeg,.png,.tif,.tiff,.dng"
        className="hidden"
        onChange={onFileChange}
      />
      <input
        ref={folderInputRef}
        type="file"
        // @ts-ignore — webkitdirectory is non-standard
        webkitdirectory=""
        multiple
        className="hidden"
        onChange={onFileChange}
      />
    </div>
  );
}
