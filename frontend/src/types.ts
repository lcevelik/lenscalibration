export interface ScoredFrame {
  index: number;
  path: string;
  found: boolean;
  corners: [number, number][];
  sharpness: number;
  coverage: number;
  angle: number | null;
  quality: 'good' | 'warn' | 'fail';
  reason: string;
  image_width: number;
  image_height: number;
}

export interface CalibrationResult {
  rms: number;
  camera_matrix: number[][];
  dist_coeffs: number[];
  fov_x: number;
  fov_y: number;
  confidence: 'excellent' | 'good' | 'marginal' | 'poor';
  per_image_errors: Array<{ path: string; error: number; outlier: boolean }>;
  used_frames: number;
  skipped_frames: number;
}

export interface BoardSettings {
  cols: number;
  rows: number;
  squareSizeMm: number;
}

export type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
