# Lens Calibration Tool

Electron + Python/FastAPI + React/TypeScript application for cinema lens calibration.

## Architecture

- **Backend**: Python/FastAPI with OpenCV for calibration (`backend/`)
- **Frontend**: React/TypeScript/Vite (`frontend/`)
- **Packaging**: Electron for desktop distribution
- **Communication**: WebSocket for real-time frame exchange

## Key Modules

### Backend
- `calib_helpers.py` — Shared calibration helpers (make_objp, confidence, is_implausible_solution, run_constrained_fallback)
- `calibrator.py` — Single-FL calibration with multi-detection fallback (SB → classic → ArUco → ChArUco → partial grid)
- `zoom_calibrator.py` — Multi-FL zoom calibration with nodal offset calculation (weighted by reprojection error, solvePnPRansac fallback, Tx/Ty/Tz export)
- `nodal_model.py` — Padé (2,1) rational model for nodal offset interpolation/extrapolation with PCHIP interpolation and pole safety checks
- `frame_scorer.py` — Frame quality scoring with pose metrics
- `pose_advisor.py` — Guided capture pose guidance (6-zone board coverage)
- `exporter.py` — Export to UE5 .ulens, OpenCV XML, STmap EXR, JSON
- `export_lens.py` — Lens file export with FL interpolation and nodal offsets
- `nodal_presets.json` — 16 cinema zoom lens presets (Angenieux, Canon CN-E, Sigma Cine, Cooke, Fujinon Premista)

### Frontend
- `App.tsx` — Main app with ZoomResultsPanel (RmsSparkline, per-FL table with Tx/Ty/Tz tooltip, nodal preset dropdown)
- `ResultPanel.tsx` — Single-FL results with DistortionCurve SVG visualization
- `GuidedCapture.tsx` — Live capture with pose guidance
- `types.ts` — Complete TypeScript type definitions

## Testing

```bash
# Backend unit tests
cd backend && python -m pytest test_nodal_model.py -v

# TypeScript type check
cd frontend && npx tsc --noEmit

# Production build
cd frontend && npx vite build
```

## Known Remaining Issues

1. **Pinhole vs entrance pupil**: `_camera_center_from_rt` uses `(-R^T @ tvec)` — gives pinhole center, not entrance pupil. For retrofocus wide-angle lenses the error can be 2-5mm.
2. **Working-distance correction**: `mean_center[2] -= working_dist_mm` assumes perfect alignment. Off-axis stepping back corrupts X/Y.
3. **No focus-distance-dependent calibration**: Distortion changes with focus distance — a near/mid/infinity sweep would be a unique differentiator.
4. **No PDF calibration report export**: Table-stakes for professional DIT workflows.
5. **No undistort before/after comparison slider**.
