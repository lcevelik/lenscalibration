# Lens Calibration

Desktop application for camera lens calibration in virtual production environments. Captures checkerboard or ArUco frames from live SDI/USB video sources or image files, computes OpenCV distortion parameters, and exports to multiple formats including UE5 `.ulens`, STmap EXR, and OpenCV XML.

---

## Features

- **Live capture** from DirectShow/MSMF capture cards and webcams (Blackmagic DeckLink, AJA, Bluefish444, Magewell, generic)
- **File-based calibration**: drag-and-drop JPEG/PNG images with per-FL focal-length groups for zoom lenses
- **Pose-guided auto-capture**: checklist of required board positions auto-captured when held steady
- **Zoom sweep calibration**: calibrate at multiple focal lengths and get a dense interpolated table
- **Sony AcuTarget ArUco support**: sparse-mode detection for DICT_4×4_50 boards (IDs 12–31)
- **Anamorphic / squeeze lens** support: apply a horizontal squeeze ratio before calibration
- **Physics-based focal-length initialisation**: solver seeded from `fl_mm × sensor_width_px` for reliable convergence at long focal lengths
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
│   ├── frame_scorer.py       # Checkerboard/ArUco detection + quality scoring
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
│           ├── FileCalibration.tsx  # File-based multi-FL zoom calibration UI
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

The dev script starts Vite on port 5173 and waits for it before launching Electron. The Python backend is spawned by Electron on a **fixed port 8765** in dev mode (random in packaged builds) and communicates via WebSocket.

### Remote access in dev mode

Vite listens on all interfaces (`0.0.0.0:5173`) and proxies `/ws` and `/thumbnail` to the backend at `127.0.0.1:8765`. Any device on the same network can open the app at `http://<host-ip>:5173` and control live capture / see the preview without needing direct access to the backend port.

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

### Zoom sweep — File-based (recommended)

1. Open the **File** tab
2. **Select your camera sensor** using the preset buttons (Sony Venice, Venice 2, Burano, Broadcast B4) or enter dimensions manually — sensor width is required for accurate initialisation
3. Click **+ Add FL group** for each focal length you want to calibrate
4. Drag-and-drop or browse for images into each group; set the FL value in mm
5. Click **Run Zoom Calibration** — only enabled once a sensor preset is selected
6. Review per-FL RMS and nodal offsets in the result table
7. Export as `.ulens` zoom via the Results panel

### Zoom sweep — Live capture

1. Open **Zoom Sweep** tab
2. For each focal length group, capture frames at that focal length, then add the group
3. Click **Run Zoom Calibration**

#### Physics-based focal-length initialisation

The zoom calibrator seeds the OpenCV solver with a physics-based initial focal length:

```
fx_init = focal_length_mm × image_width_px / sensor_width_mm
```

This is critical for telephoto focal lengths (e.g. 100mm on a 36mm sensor → fx ≈ 5333px) where the default heuristic of `max(width, height)` would cause the solver to diverge. The sensor width must be provided for this to take effect.

#### Nodal Stability With Large Charts

The zoom nodal solver uses **scale-normalized object points** before running
`calibrateCamera` / `solvePnP`, then converts translation back to millimetres.
This improves numerical conditioning when chart dimensions are very large
(telephoto workflows, large stage boards).

### File import (single FL)

Drag and drop JPEG/PNG images onto the **File** tab (single group) to calibrate a prime or fixed-zoom lens from still images.

---

## Sensor Presets

The following Sony sensor presets are available in the **File Calibration** tab and the **Live Capture** tab:

| Preset | Width (mm) | Height (mm) |
|---|---|---|
| Venice FF 36×24 | 36.0 | 24.0 |
| Venice S35 24×13 | 24.0 | 13.0 |
| Venice S16 14×7.9 | 14.0 | 7.9 |
| Venice 2 / Burano FF 35.9×24 | 35.9 | 24.0 |
| Venice 2 S35 23.6×13.3 | 23.6 | 13.3 |
| HDC-series 2/3" B4 9.59×5.39 | 9.59 | 5.39 |
| HDC-F5500 / HDC-3500 4K 2/3" | 9.59 | 5.39 |
| HDC-5500 / HDW-series 2/3" | 9.59 | 5.39 |

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

Set `squeezeRatio > 1.0` (e.g. `2.0` for 2× anamorphic) to enable squeeze compensation. The default for spherical lenses is `1.0`.

- Corner coordinates are scaled horizontally by the squeeze ratio before calibration
- Calibration is performed in de-squeezed pixel space (`width × squeezeRatio`)
- STmap EXR output width is `width × squeezeRatio`
- `.ulens` export includes `squeezeRatio` and `lensType: anamorphic` metadata

---

## Camera Support

Detection priority on Windows:

1. **pygrabber** (ICreateDevEnum/IEnumMoniker) — most reliable for capture cards
2. **PowerShell WMI** (`Get-PnpDevice` Camera/Media class) — fallback
3. **OpenCV scan** — indices 0–11 with `CAP_DSHOW`

Capture opens with `CAP_DSHOW` first (required for SDI cards), falls back to `CAP_MSMF`, then default backend. SDI cards lock to signal resolution; the actual resolution is read back after opening.

The device list is filtered to **IO cards and webcams only** — microphones, headsets, audio interfaces, scanners, and virtual cameras (OBS Virtual Camera, Snap Camera, etc.) are automatically excluded.

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
| `calibrate` | Single-FL calibration from scored frames |
| `calibrate_zoom` | Multi-FL zoom calibration |
| `export` | Export results to file |
| `get_undistort_preview` | Generate undistorted image pair |
| `list_devices` | Enumerate capture devices |

The client reconnects automatically with exponential backoff (1 s → 2 s → 4 s → 8 s → 16 s cap) on disconnect.

---

## Quality Thresholds

Quality grades are board-type aware. Sparse ArUco boards (e.g. Sony AcuTarget) have fewer detection points than dense checkerboards, so RMS naturally runs higher.

### Dense checkerboard

| Grade | RMS threshold |
|---|---|
| Excellent | < 0.3 px |
| Good | < 0.5 px |
| Marginal | < 1.0 px |
| Poor | ≥ 1.0 px |

### Sparse ArUco board (e.g. Sony AcuTarget)

| Grade | RMS threshold |
|---|---|
| Excellent | < 0.7 px |
| Good | < 1.5 px |
| Marginal | < 3.0 px |
| Poor | ≥ 3.0 px |

### Frame scoring

| Metric | Threshold |
|---|---|
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



