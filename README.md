# Lens Calibration

Desktop application for camera lens calibration in virtual production environments. Captures checkerboard frames from live SDI/USB video sources or image files, computes OpenCV distortion parameters, and exports to multiple formats including UE5 `.ulens`, STmap EXR, and OpenCV XML.

---

## Features

- **Live capture** from DirectShow/MSMF capture cards and webcams (Blackmagic DeckLink, AJA, Bluefish444, Magewell, generic)
- **Pose-guided auto-capture**: checklist of required board positions auto-captured when held steady
- **Zoom sweep calibration**: calibrate at multiple focal lengths and get a dense interpolated table
- **Anamorphic / squeeze lens** support: apply a horizontal squeeze ratio before calibration
- **Export formats**: OpenCV XML, JSON, STmap EXR, UE5 `.ulens` (single FL and zoom sweep)
- **Undistort preview**: side-by-side comparison with blob-URL memory management

---

## Architecture

```
lenscalibration/
├── backend/                  # Python FastAPI server
│   ├── main.py               # WebSocket dispatcher + HTTP endpoints
│   ├── calibrator.py         # Single-FL OpenCV calibration
│   ├── zoom_calibrator.py    # Multi-FL zoom sweep + PCHIP interpolation
│   ├── exporter.py           # XML / JSON / EXR / .ulens exporters
│   ├── frame_scorer.py       # Checkerboard detection + quality scoring
│   ├── live_capture.py       # Async live capture + pose-guided auto-capture
│   ├── capture_device.py     # Device enumeration (pygrabber / PowerShell / OpenCV)
│   ├── pose_advisor.py       # Pose checklist evaluation
│   └── requirements.txt
├── electron/
│   ├── main.js               # Electron main process (window + IPC + backend lifecycle)
│   ├── preload.js            # contextBridge API surface
│   └── launch.js             # Dev launcher
├── frontend/
│   └── src/
│       ├── App.tsx           # Root component + WebSocket client
│       └── components/
│           ├── GuidedCapture.tsx    # Live capture UI
│           ├── ZoomSweep.tsx        # Zoom sweep UI
│           ├── ResultPanel.tsx      # Calibration results display
│           ├── UndistortPreview.tsx # Side-by-side undistort viewer
│           ├── FrameGrid.tsx        # Captured frame thumbnails
│           ├── CoverageMap.tsx      # Board coverage heat-map
│           ├── PoseDiagram.tsx      # Pose checklist diagram
│           ├── DropZone.tsx         # File-drop import
│           └── BoardPresetSelector.tsx
└── package.json              # Root — Electron + build scripts
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Node.js | 18+ |
| Python | 3.10+ |
| pip packages | see `backend/requirements.txt` |

**Python packages** (`pip install -r backend/requirements.txt`):
```
fastapi
uvicorn[standard]
opencv-python
numpy
scipy
websockets
openexr
imath
pygrabber
```

> `openexr` and `imath` are only required for STmap EXR export.
> `pygrabber` is only required on Windows for DirectShow device enumeration.

---

## Development Setup

```bash
# 1. Install Node dependencies (root + Electron build tools)
npm install

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Start in development mode (Vite + Electron, hot-reload)
npm run dev
```

The dev script starts Vite on port 5173 and waits for it before launching Electron. The Python backend is spawned by Electron on a random available port and communicates via WebSocket.

---

## Building

```bash
# Build frontend then package with electron-builder
npm run build
```

Packaged outputs go to `dist-electron/`. The backend Python files are bundled as `extraResources` — you must ensure Python is installed on the target machine (or bundle a Python distribution).

---

## Calibration Workflow

### Single focal length (prime or fixed zoom)

1. Select **Capture** tab, choose video device and board preset (cols × rows × square size mm)
2. Hold the calibration board in each required pose — the checklist auto-captures when the board is held steady for 0.5 s
3. Once the checklist is complete (or manually when enough frames are captured), click **Calibrate**
4. Review results in the **Results** panel: RMS reprojection error, per-image outliers, distortion coefficients
5. Export via **Export** tab

### Zoom sweep (zoom lenses)

1. Open **Zoom Sweep** tab
2. For each focal length group, capture frames at that focal length, then add the group
3. Click **Run Zoom Calibration** — each FL is calibrated independently; nodal offsets are computed relative to the best-RMS focal length
4. Export as `.ulens` (zoom) — produces a dense interpolated table (up to 100 rows between measured FLs via PCHIP)

### File import

Drag and drop JPEG/PNG images onto the **Drop Zone** panel to use still images instead of live capture.

---

## Export Formats

### OpenCV XML
Standard `cv2.FileStorage` format. Compatible with Ventuz FreeD and other tools that read OpenCV calibration files.

### JSON
Human-readable calibration with metadata: `fx`, `fy`, `cx`, `cy`, distortion coefficients, FOV, calibration date.

### STmap EXR
32-bit float UV remap map. `R` channel = U (normalised horizontal), `G` channel = V (normalised vertical). Use in Nuke, Resolve, or any compositor that supports STmap-based undistortion. Anamorphic lenses produce a de-squeezed output width (`width × squeeze_ratio`).

### UE5 .ulens (single FL)
CameraCalibration plugin `SphericalLensModel` format. Static lens with `ZoomEncoder = 0`, `FocusEncoder = 0`. Fx/Fy/Cx/Cy are normalised by image dimensions.

### UE5 .ulens (zoom sweep)
Multi-entry zoom table. `ZoomEncoder` is normalised `[0, 1]` across the captured FL range. `NodalOffset Tz` carries the optical-centre Z-shift in mm (OpenCV coordinates, relative to best-RMS FL). When interpolated rows are included, UE5 gets a fine-grained table that avoids linear-interpolation artefacts across large FL gaps.

---

## Anamorphic / Squeeze Lenses

Set `squeezeRatio > 1.0` (e.g. `2.0` for 2× anamorphic) to enable squeeze compensation:
- Corner coordinates are scaled horizontally by the squeeze ratio before calibration
- Calibration is performed in de-squeezed pixel space (`width × squeezeRatio`)
- STmap EXR output width is `width × squeezeRatio`
- `.ulens` export includes `squeezeRatio` and `lensType: anamorphic` metadata

---

## Camera Support

Detection priority on Windows:

1. **pygrabber** (ICreateDevEnum/IEnumMoniker) — most reliable for capture cards
2. **PowerShell WMI** (`Get-PnpDevice` Camera/Image/Media class) — fallback
3. **OpenCV scan** — indices 0–11 with `CAP_DSHOW`

Capture opens with `CAP_DSHOW` first (required for SDI cards), falls back to `CAP_MSMF`, then default backend. SDI cards lock to signal resolution; the actual resolution is read back after opening.

Recognised brands: Blackmagic DeckLink, AJA, Bluefish444, Magewell, Datapath, DELTACAST.

---

## WebSocket Protocol

The frontend communicates with the Python backend over a single WebSocket connection. All messages are JSON. Key actions:

| Action (client → server) | Description |
|---|---|
| `start_preview` | Start live preview stream |
| `stop_preview` | Stop preview |
| `start_live_capture` | Begin pose-guided capture session |
| `manual_capture` | Force-capture current frame |
| `stop_live_capture` | End capture, receive scored frames |
| `run_calibration` | Calibrate from scored frames |
| `run_zoom_calibration` | Multi-FL zoom calibration |
| `export` | Export results to file |
| `get_undistort_preview` | Generate undistorted image pair |
| `list_devices` | Enumerate capture devices |

The client reconnects automatically with exponential backoff (1 s → 2 s → 4 s → 8 s → 16 s cap) on disconnect.

---

## Quality Thresholds

| Metric | Threshold |
|---|---|
| RMS reprojection error — excellent | < 0.3 px |
| RMS reprojection error — good | < 0.5 px |
| RMS reprojection error — marginal | < 1.0 px |
| Frame sharpness (hard fail) | Laplacian variance < 30 |
| Frame sharpness (warn) | Laplacian variance < 80 |
| Outlier detection | Tukey IQR: Q3 + 1.5 × IQR |

---

## Security Notes

- The thumbnail HTTP endpoint and undistort handler validate all file paths: `.` traversal rejected, `os.path.realpath` checked, must be an existing file.
- Electron `show-save-dialog` IPC whitelists only `defaultPath` and `filters` from client options.
- CORS is restricted to `localhost` origins only.

---

## License

MIT
