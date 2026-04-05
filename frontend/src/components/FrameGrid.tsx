import type { ScoredFrame } from '../types';

const QUALITY_BADGE: Record<string, string> = {
  good: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
  warn: 'bg-yellow-500/20  text-yellow-400  border-yellow-500/40',
  fail: 'bg-red-500/20     text-red-400     border-red-500/40',
};

interface Props {
  frames: ScoredFrame[];
  excluded: Set<string>;
  onToggle: (path: string) => void;
  backendPort: number;
}

function filename(p: string) {
  return p.replace(/\\/g, '/').split('/').pop() ?? p;
}

export default function FrameGrid({ frames, excluded, onToggle, backendPort }: Props) {
  if (frames.length === 0) return null;

  const included = frames.filter(f => !excluded.has(f.path));
  const thumbUrl = (path: string) =>
    import.meta.env.DEV
      ? `/thumbnail?path=${encodeURIComponent(path)}&width=240`
      : `http://127.0.0.1:${backendPort}/thumbnail?path=${encodeURIComponent(path)}&width=240`;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Frames
        </h3>
        <span className="text-xs text-slate-500">
          {included.length} selected / {frames.length} total
        </span>
      </div>

      <div className="grid grid-cols-4 gap-1.5 max-h-80 overflow-y-auto pr-1">
        {frames.map((frame) => {
          const isExcluded = excluded.has(frame.path);
          return (
            <button
              key={frame.path}
              onClick={() => onToggle(frame.path)}
              title={`${filename(frame.path)}\n${frame.reason}`}
              className="relative group rounded-lg overflow-hidden border border-slate-700 hover:border-slate-500 transition-colors focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {/* Thumbnail */}
              <div className="aspect-video bg-slate-900">
                {frame.found ? (
                  <img
                    src={thumbUrl(frame.path)}
                    alt={filename(frame.path)}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs">
                    No board
                  </div>
                )}
              </div>

              {/* Quality badge */}
              <div className="absolute top-1 left-1">
                <span className={`text-[9px] font-bold px-1 py-0.5 rounded border uppercase ${QUALITY_BADGE[frame.quality]}`}>
                  {frame.quality}
                </span>
              </div>

              {/* Excluded overlay */}
              {isExcluded && (
                <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                    Excluded
                  </span>
                </div>
              )}

              {/* Hover info */}
              <div className="absolute bottom-0 inset-x-0 bg-slate-900/80 px-1 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <p className="text-[9px] text-slate-300 truncate">{filename(frame.path)}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
