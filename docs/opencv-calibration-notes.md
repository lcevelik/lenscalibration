# OpenCV Lens Calibration — Technical Reference

## What OpenCV Does

OpenCV provides the mathematical solvers — the codebase is the orchestration layer. Key functions:

| Function | Purpose |
|---|---|
| `cv2.findChessboardCornersSB` / `findChessboardCorners` | Detects grid corners in the image |
| `cv2.cornerSubPix` | Refines corner positions to sub-pixel accuracy (critical for precision) |
| `cv2.calibrateCamera` | Solves a nonlinear optimization (Levenberg-Marquardt) to find K, distortion coefficients, and per-image poses |
| `cv2.Rodrigues` | Converts rotation vector → 3×3 matrix |
| `cv2.projectPoints` | Reprojects 3D points back to image to measure reprojection error |
| `cv2.solvePnP` | When intrinsics are already known, solves only the pose — used as fallback for telephoto focal lengths |
| `cv2.initUndistortRectifyMap` | Builds UV remap tables for STmap export |

### What `calibrateCamera` Is Actually Solving

```
Minimize: Σᵢ Σⱼ ‖ project(Rᵢ, tᵢ, K, d, Pⱼ) − pᵢⱼ ‖²
```

- `K` = camera matrix (focal length + principal point)
- `d` = distortion coefficients (k1, k2, k3, p1, p2) — Brown-Conrady model
- `(Rᵢ, tᵢ)` = pose for each image
- All parameters solved simultaneously via Levenberg-Marquardt

---

## Is OpenCV the Only Way?

No — but it is the most practical choice.

| Approach | Pros | Cons |
|---|---|---|
| **OpenCV** | Battle-tested, fast, well-supported | Black-box solver, limited control |
| **Ceres Solver** (C++) | More control, custom residuals, bundle adjustment | Complex, C++-only, requires reimplementing the camera model |
| **COLMAP** | Great for SfM | Not designed for calibration charts |
| **Matlab Camera Calibrator** | Excellent GUI, validated math | Paid, not embeddable |
| **Manual / custom solver** | Full control | Reimplementing Zhang's method is significant work |
| **Factory calibration** | No images needed | Rarely accurate enough for VFX |

The codebase already wraps OpenCV intelligently — physics-based initialization, outlier removal, constrained flags for telephoto. That is exactly what you would do with a custom solver anyway.

---

## Image Count Requirements

| Count | Status |
|---|---|
| **3 frames** | Minimum enforced (`calibrator.py:155`) |
| **10 frames** | Recommended — guided 10-pose checklist (`pose_advisor.py:29-100`) |
| **10–15 frames** | Best practice — outlier removal re-runs calibration after dropping bad frames, so spare frames help |

For **zoom lenses**: minimum 3 usable frames *per focal length stop*. More stops = better Padé model fit (needs ≥ 4 points for a meaningful curve).

### Why 10 Poses

Each pose constrains different parameters:

| Pose | What it constrains |
|---|---|
| Center flat | Principal point baseline, radial reference |
| Center tilted H/V | Higher-order radial (k2, k3) |
| Corners (4×) | Field-edge distortion, principal point offset |
| Edge positions | Independent vertical distortion |
| Close-up | Focal length accuracy |

---

## What You Need for Best Results

### Hardware
- Flat, rigid chart — any flex introduces false distortion
- Diffuse, even lighting — no hotspots, no reflections
- Chart fills ≥ 50% of frame in some shots; reaches corners in others

### Capture
- Sharp focus — Laplacian variance gate is > 15, but aim for > 100
- All 10 poses per focal length
- For zoom calibration: measure at enough stops that the curve is well-sampled (not just two endpoints)

### Settings
- `square_size_mm` must match your physical chart exactly — a 1% size error causes ~1% focal length error
- `working_dist_mm` must be set correctly for zoom calibration — the nodal Z subtraction at `zoom_calibrator.py:427` depends on it

---

## Nodal Offset Calculation

### What It Is

The entrance pupil (optical center) position along the optical axis (Z) varies with focal length in zoom lenses. The pipeline measures this shift per focal length and fits a model for extrapolation.

### How It Works

**Step 1 — Per-frame optical center extraction** (`zoom_calibrator.py:418-430`):
```python
R, _ = cv2.Rodrigues(rvec)          # rotation vector → 3×3 matrix
center = -R.T @ tvec                # optical center in world coords [x, y, z] mm
```

**Step 2 — Average across frames at each FL, then subtract working distance**:
```python
mean_center = np.mean(centers, axis=0)
if working_dist_mm > 0:
    mean_center[2] -= working_dist_mm
```

**Step 3 — Relative offset from reference FL** (`zoom_calibrator.py:549-557`):
```python
# Reference = FL with lowest RMS
nodal_offset[fl] = center_z[fl] - center_z[ref_fl]
```

**Step 4 — Model fitting** (`nodal_model.py`):

| Points | Model |
|---|---|
| 2 | Linear polynomial |
| 3 | Quadratic polynomial (exact solution) |
| ≥ 4 | Padé (2,1) rational: `N(f) = (a₀ + a₁f + a₂f²) / (1 + b₁f)` |

Padé falls back to quadratic if fitting fails.

**Step 5 — Extrapolation** (`zoom_calibrator.py:806-816`):
- Inside measured range: PCHIP interpolation
- Outside measured range: Padé model prediction
- No model available: linear extrapolation from boundary

### Why Padé (2,1)?

Zoom lens entrance pupils shift smoothly with focal length. Mechanically compensated zooms follow approximately linear behavior at short/medium ranges, but the rational form of Padé captures asymptotic behavior at the extremes that a polynomial misses. It matches real optical designs well.

### Potential Failure Modes

| Risk | Symptom | Fix |
|---|---|---|
| `working_dist_mm` wrong or unset | Absolute Z offset wrong; relative offsets still correct | Measure carefully, or use relative-only export |
| Only 2 focal lengths measured | Falls back to linear — no curve shape | Get ≥ 4 focal lengths for a meaningful Padé fit |
| High RMS at one FL | That FL's optical center estimate is noisy | Pipeline uses lowest-RMS FL as reference; high-RMS FLs fall back to `solvePnP` with best-known K |
| Camera moved between FLs | Optical centers will be inconsistent | Lock camera on tripod, zoom only — chart must be stationary |
| Zoom breathing | Working distance changes as you zoom | Set `working_dist_mm` per FL if breathing is significant |

The `solvePnP` fallback path (`zoom_calibrator.py:503`) is a good safety net — if a telephoto FL can't fully calibrate on its own, it borrows K from the best FL and solves only for pose, which is sufficient to extract nodal Z.

---

## Should You Use Ceres Solver Instead?

**No — not unless you hit a concrete wall OpenCV can't solve.**

### Where Ceres Actually Wins

| Scenario | Why Ceres helps |
|---|---|
| Bundle adjustment across multiple cameras simultaneously | OpenCV can't do this; Ceres handles thousands of parameters |
| Custom distortion models beyond Brown-Conrady | You control every residual term |
| Joint optimization across all zoom positions in one solve | OpenCV calibrates each FL independently; Ceres could share parameters across FLs |
| Hard constraints with precise penalties | OpenCV uses coarse flags; Ceres uses soft/hard penalty terms |

### Why It Is Not Worth It Here

- OpenCV's LM solver converges to the same minimum Ceres would, given the same data and initialization
- The outlier removal, physics-based initialization, and constrained fallbacks already handle the cases where a naive solver fails
- The RMS targets (< 0.3 px = excellent) are well within what OpenCV achieves

### The One Genuine Improvement Ceres Would Offer

A **single joint solve** across all focal lengths, where distortion coefficients and optical center curve are *shared parameters* — making the zoom model more physically coherent. Right now the pipeline calibrates each FL independently and fits a Padé model after the fact. Ceres could do it in one shot.

That is a significant rewrite. Only worth it if nodal accuracy is provably insufficient with the current approach.

### The Right Trigger to Reconsider

- Nodal offsets are noisy and the Padé fit is poor even with good capture data
- You need to calibrate a multi-camera rig jointly
- You need a physically-motivated distortion model beyond Brown-Conrady
- You are hitting accuracy walls below ~0.3 px RMS that cannot be explained by data quality

---

## Key File Reference

| File | Lines | Responsibility |
|---|---|---|
| `backend/calibrator.py` | ~378 | Single-FL OpenCV calibration, RMS validation, outlier filtering |
| `backend/zoom_calibrator.py` | ~858 | Multi-FL calibration, optical center extraction, nodal offset, interpolation |
| `backend/frame_scorer.py` | ~410 | Board detection (6 methods), ArUco/ChArUco, quality scoring |
| `backend/nodal_model.py` | ~147 | Padé/poly model fitting and extrapolation |
| `backend/pose_advisor.py` | ~656 | 10-pose checklist design, pose matching logic |
| `backend/exporter.py` | ~599 | XML / JSON / EXR / UE5 .ulens export |
| `backend/live_capture.py` | ~592 | WebSocket streaming, pose-guided auto-capture |
| `backend/main.py` | ~429 | FastAPI endpoints, WebSocket dispatcher |
