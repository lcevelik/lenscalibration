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
  /** Set when only a sub-region of the chart was detected (telephoto overflow). */
  partial_grid_size?: [number, number] | null;
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

export interface LensSettings {
  lensType: 'spherical' | 'anamorphic';
  squeezeRatio: number; // 1.0 for spherical; 1.33, 1.5, 1.8, 2.0 for anamorphic
}

export interface CameraSettings {
  lensName: string;
  sensorWidthMm: string;   // string so inputs work naturally; parse on use
  sensorHeightMm: string;
  nodalPreset: string;     // key from nodal_presets.json, e.g. "fujinon-premista-28-100"
}

export type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

