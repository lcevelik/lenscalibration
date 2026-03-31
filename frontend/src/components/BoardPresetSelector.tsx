import { useState } from 'react';
import type { BoardSettings } from '../types';

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

interface BoardPreset {
  id:           string;
  label:        string;
  description:  string;
  cols:         number;
  rows:         number;
  squareSizeMm: number;
}

export const BOARD_PRESETS: BoardPreset[] = [
  {
    id:           'sony_ocellus',
    label:        'Sony Ocellus Chart',
    description:  '9 × 6 inner corners · 50 mm squares (5 cm)',
    cols:         9,
    rows:         6,
    squareSizeMm: 50,
  },
  {
    id:           'custom',
    label:        'Custom',
    description:  'Enter your own board dimensions',
    cols:         9,
    rows:         6,
    squareSizeMm: 25,
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  boardSettings: BoardSettings;
  onBoardChange: (s: BoardSettings) => void;
  disabled?: boolean;
}

export default function BoardPresetSelector({ boardSettings, onBoardChange, disabled }: Props) {
  const [presetId, setPresetId] = useState<string>('sony_ocellus');

  const selectPreset = (id: string) => {
    setPresetId(id);
    const preset = BOARD_PRESETS.find(p => p.id === id);
    if (preset && id !== 'custom') {
      onBoardChange({
        cols:         preset.cols,
        rows:         preset.rows,
        squareSizeMm: preset.squareSizeMm,
      });
    }
  };

  const isCustom = presetId === 'custom';
  const activePreset = BOARD_PRESETS.find(p => p.id === presetId);

  return (
    <div className="space-y-2">
      {/* Preset buttons */}
      <div className="flex gap-2 flex-wrap">
        {BOARD_PRESETS.map(preset => (
          <button
            key={preset.id}
            type="button"
            disabled={disabled}
            onClick={() => selectPreset(preset.id)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors disabled:opacity-50 ${
              presetId === preset.id
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600 hover:border-slate-500'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Description / values for active preset */}
      {!isCustom && activePreset && (
        <p className="text-[11px] text-slate-500">{activePreset.description}</p>
      )}

      {/* Custom inputs — only shown when Custom is selected */}
      {isCustom && (
        <div className="grid grid-cols-3 gap-2 pt-1">
          {(
            [
              { label: 'Inner cols',  key: 'cols',         min: 2,  max: 30  },
              { label: 'Inner rows',  key: 'rows',         min: 2,  max: 30  },
              { label: 'Square mm',   key: 'squareSizeMm', min: 1,  max: 500 },
            ] as const
          ).map(({ label, key, min, max }) => (
            <label key={key} className="flex flex-col gap-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
              <input
                type="number"
                min={min}
                max={max}
                value={boardSettings[key]}
                disabled={disabled}
                onChange={e => onBoardChange({ ...boardSettings, [key]: Number(e.target.value) })}
                className="bg-slate-700 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-200 text-center focus:outline-none focus:border-blue-500 w-full disabled:opacity-50"
              />
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
