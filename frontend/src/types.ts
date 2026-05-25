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
  board_type?: string;
  detection_type?: string;
  obj_points?: [number, number, number][];
  pose_metrics?: {
    board_center_px: [number, number] | null;
    board_center_frac: [number, number] | null;
    board_fill_frac: number;
    roll_deg: number;
    vertical_tilt_deg: number;
    yaw_deg: number;
    quadrant: string;
  };
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
  squeeze_ratio?: number;
  lens_type?: string;
  error?: string | null;
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
  nodalPreset: string;     // key from nodal_presets.json
}

export type ConnStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// ── Multi-FL (zoom) calibration ──────────────────────────────────────────────

export interface ZoomFlResult {
  focal_length_mm: number;
  rms: number | null;
  fx_px?: number;
  fy_px?: number;
  cx_px?: number;
  cy_px?: number;
  focal_length_computed_mm?: number;
  dist_coeffs?: number[];
  camera_matrix?: number[][];
  optical_center_world?: number[];
  used_frames?: number;
  per_image_errors?: Array<{ path: string; error: number; outlier: boolean }>;
  confidence?: string;
  error?: string | null;
  pose_only?: boolean;
  warning?: string;
  interpolated?: boolean;
  extrapolated?: boolean;
  nodal_offset_z_mm?: number;
  nodal_offset_x_mm?: number;
  nodal_offset_y_mm?: number;
}

export interface NodalOffsets {
  [flKey: string]: { tx: number; ty: number; tz: number };
}

export interface NodalModel {
  model: "poly1" | "poly2" | "pade21";
  coeffs: number[];
  fl_min: number;
  fl_max: number;
  fit_rms_mm: number;
}

export interface ZoomCalibrationResult {
  success: boolean;
  error: string | null;
  fl_results: ZoomFlResult[];
  fl_interpolated?: ZoomFlResult[];
  nodal_offsets_mm: NodalOffsets;
  nodal_model?: NodalModel | null;
  squeeze_ratio: number;
  lens_type: string;
}

// ── Export ───────────────────────────────────────────────────────────────────

export interface ExportResult {
  success: boolean;
  output_path?: string;
  error?: string;
  fl_count?: number;
  squeeze_ratio?: number;
  nodal_model?: string;
  nodal_offsets_count?: number;
  board_type?: string;
  board_size?: string;
  square_size_mm?: number;
  fl_interpolated_count?: number;
}

// ── Live capture ─────────────────────────────────────────────────────────────

export interface CapturedFrame {
  filename: string;
  path: string;
  score: ScoredFrame;
  timestamp: number;
}

export interface CaptureProgress {
  captured: number;
  target: number;
  remaining: number;
}

// ── Guidance ─────────────────────────────────────────────────────────────────

export interface GuidanceStep {
  id: string;
  label: string;
  description: string;
  target_yaw: number;
  target_tilt: number;
  quadrant: string;
  icon: string;
}

export interface PoseAdvice {
  guidance_steps: GuidanceStep[];
  current_quadrant: string;
  current_yaw: number;
  current_tilt: number;
  board_fill: number;
  recommendations: string[];
}
