import argparse
import asyncio
import atexit
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect

from calibrator import run_calibration
from capture_device import enumerate_capture_devices, open_capture, read_actual_size, read_actual_fps
from exporter import export_json, export_opencv_xml, export_stmap_exr, export_ue5_ulens, export_ue5_ulens_zoom
from zoom_calibrator import run_zoom_calibration
from frame_scorer import score_frame
from live_capture import run_live_capture, run_preview

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # Restrict to localhost origins only — this backend is not a public API
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Size the pool so CPU-bound calibration tasks don't starve preview I/O
_executor = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 1) + 4))
atexit.register(lambda: _executor.shutdown(wait=False))


def _is_safe_path(path: str) -> bool:
    """Return True if path points to a regular image file with no traversal."""
    try:
        resolved = os.path.realpath(os.path.abspath(path))
        # Reject obvious traversal sequences even before realpath
        if ".." in path:
            return False
        # Allow any absolute path that resolves to a real file; the Electron
        # front-end always sends paths it obtained from a file-picker dialog.
        return os.path.isfile(resolved)
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def _make_thumbnail(path: str, width: int) -> Optional[bytes]:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    new_w = min(width, w)
    new_h = int(h * new_w / w)
    thumb = cv2.resize(img, (new_w, new_h))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return bytes(buf)


@app.get("/thumbnail")
async def get_thumbnail(path: str, width: int = 200):
    if not _is_safe_path(path):
        raise HTTPException(status_code=403, detail="Access denied")
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _make_thumbnail, path, width)
    if data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------

def _clamp_board(cols: int, rows: int) -> tuple[int, int]:
    """Clamp board dimensions to a safe range."""
    return max(2, min(cols, 25)), max(2, min(rows, 25))


async def _handle_score_frames(websocket: WebSocket, message: dict) -> None:
    paths: list = message.get("paths", [])
    if not isinstance(paths, list):
        await websocket.send_text(json.dumps({"error": "paths must be a list"}))
        return
    board_cols: int = int(message.get("board_cols", 9))
    board_rows: int = int(message.get("board_rows", 6))
    board_cols, board_rows = _clamp_board(board_cols, board_rows)
    checkerboard_size = (board_cols, board_rows)
    loop = asyncio.get_event_loop()
    for index, path in enumerate(paths):
        result = await loop.run_in_executor(_executor, score_frame, path, checkerboard_size)
        await websocket.send_text(
            json.dumps({"action": "frame_result", "index": index, "path": path, **result})
        )
    await websocket.send_text(json.dumps({"action": "score_frames_done", "total": len(paths)}))


async def _handle_calibrate(websocket: WebSocket, message: dict) -> None:
    scored_frames: list = message.get("scored_frames", [])
    board_cols: int = int(message.get("board_cols", 9))
    board_rows: int = int(message.get("board_rows", 6))
    board_cols, board_rows = _clamp_board(board_cols, board_rows)
    square_size_mm: float = max(1.0, min(float(message.get("square_size_mm", 25.0)), 500.0))
    image_size: tuple = tuple(message.get("image_size", [1920, 1080]))
    squeeze_ratio: float = max(1.0, min(float(message.get("squeeze_ratio", 1.0)), 4.0))
    usable_count = sum(1 for f in scored_frames if f.get("quality") != "fail")
    await websocket.send_text(json.dumps({
        "action": "calibrate_progress",
        "status": "running",
        "message": f"Running calibration on {usable_count} usable frames…",
    }))
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, run_calibration, scored_frames, board_cols, board_rows, square_size_mm, image_size, squeeze_ratio,
    )
    await websocket.send_text(json.dumps({"action": "calibrate_result", **result}))


async def _handle_calibrate_zoom(websocket: WebSocket, message: dict) -> None:
    fl_groups:      list  = message.get("fl_groups", [])
    board_cols:     int   = int(message.get("board_cols", 9))
    board_rows:     int   = int(message.get("board_rows", 6))
    board_cols, board_rows = _clamp_board(board_cols, board_rows)
    square_size_mm: float = max(1.0, min(float(message.get("square_size_mm", 25.0)), 500.0))
    image_size:     tuple = tuple(message.get("image_size", [1920, 1080]))
    sensor_width_mm:  float = max(0.0, float(message.get("sensor_width_mm", 0.0)))
    sensor_height_mm: float = max(0.0, float(message.get("sensor_height_mm", 0.0)))
    squeeze_ratio:    float = max(1.0, min(float(message.get("squeeze_ratio", 1.0)), 4.0))

    total_frames = sum(len(g.get("frames", [])) for g in fl_groups)
    await websocket.send_text(json.dumps({
        "action":  "zoom_calibrate_progress",
        "status":  "running",
        "message": f"Running zoom calibration across {len(fl_groups)} focal lengths ({total_frames} frames)…",
    }))
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, run_zoom_calibration,
        fl_groups, board_cols, board_rows, square_size_mm, image_size,
        sensor_width_mm, sensor_height_mm, squeeze_ratio,
    )
    await websocket.send_text(json.dumps({"action": "zoom_calibrate_result", **result}))


_EXPORT_FORMATS = {
    "opencv_xml":     export_opencv_xml,
    "json":           export_json,
    "stmap_exr":      export_stmap_exr,
    "ue5_ulens":      export_ue5_ulens,
    "ue5_ulens_zoom": export_ue5_ulens_zoom,
}


async def _handle_export(websocket: WebSocket, message: dict) -> None:
    fmt: str = message.get("format", "json")
    output_path: str = message.get("output_path", f"calibration.{fmt}")
    camera_matrix = message.get("camera_matrix")
    dist_coeffs   = message.get("dist_coeffs")
    fov_x: float  = float(message.get("fov_x", 0.0))
    fov_y: float  = float(message.get("fov_y", 0.0))
    rms: float    = float(message.get("rms", 0.0))
    image_size    = tuple(message.get("image_size", [1920, 1080]))
    metadata      = message.get("metadata", {})
    if fmt not in _EXPORT_FORMATS:
        await websocket.send_text(json.dumps({
            "action": "export_result", "success": False,
            "error": f"Unknown format '{fmt}'. Choose from: {list(_EXPORT_FORMATS)}",
        }))
        return
    squeeze_ratio: float = float(message.get("squeeze_ratio", 1.0))
    lens_type: str = message.get("lens_type", "spherical")

    loop = asyncio.get_event_loop()
    if fmt == "json":
        fn = lambda: export_json(output_path, camera_matrix, dist_coeffs, fov_x, fov_y, rms, metadata,
                                 squeeze_ratio=squeeze_ratio, lens_type=lens_type)
    elif fmt == "opencv_xml":
        fn = lambda: export_opencv_xml(output_path, camera_matrix, dist_coeffs, fov_x, fov_y, image_size)
    elif fmt == "stmap_exr":
        fn = lambda: export_stmap_exr(output_path, camera_matrix, dist_coeffs, image_size,
                                      squeeze_ratio=squeeze_ratio)
    elif fmt == "ue5_ulens_zoom":
        fl_results       = message.get("fl_results", [])
        fl_interpolated  = message.get("fl_interpolated")          # may be None
        nodal_offsets_mm = message.get("nodal_offsets_mm", {})
        lens_name        = message.get("lens_name", "Lens")
        sensor_width_mm  = float(message.get("sensor_width_mm",  0.0))
        sensor_height_mm = float(message.get("sensor_height_mm", 0.0))
        fn = lambda: export_ue5_ulens_zoom(
            output_path, fl_results, image_size, nodal_offsets_mm,
            lens_name=lens_name,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            squeeze_ratio=squeeze_ratio,
            lens_type=lens_type,
            fl_interpolated=fl_interpolated,
        )
    else:
        lens_name        = message.get("lens_name", "Lens")
        sensor_width_mm  = float(message.get("sensor_width_mm", 0.0))
        sensor_height_mm = float(message.get("sensor_height_mm", 0.0))
        fn = lambda: export_ue5_ulens(
            output_path, camera_matrix, dist_coeffs, fov_x, fov_y, image_size,
            lens_name=lens_name,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            squeeze_ratio=squeeze_ratio,
            lens_type=lens_type,
        )
    result = await loop.run_in_executor(_executor, fn)
    await websocket.send_text(json.dumps({"action": "export_result", "format": fmt, **result}))


def _do_undistort(path: str, camera_matrix: list, dist_coeffs: list) -> dict:
    if not _is_safe_path(path):
        return {"success": False, "error": "Access denied"}
    img = cv2.imread(path)
    if img is None:
        return {"success": False, "error": f"Cannot read image: {path}"}
    cm = np.array(camera_matrix, dtype=np.float64)
    dc = np.array(dist_coeffs,   dtype=np.float64)
    undistorted = cv2.undistort(img, cm, dc)
    _, orig_buf  = cv2.imencode(".jpg", img,         [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, undis_buf = cv2.imencode(".jpg", undistorted, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        "success":     True,
        "original":    base64.b64encode(orig_buf).decode(),
        "undistorted": base64.b64encode(undis_buf).decode(),
        "path":        path,
    }


async def _handle_preview_undistort(websocket: WebSocket, message: dict) -> None:
    path          = message.get("path", "")
    camera_matrix = message.get("camera_matrix", [])
    dist_coeffs   = message.get("dist_coeffs",   [])
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _do_undistort, path, camera_matrix, dist_coeffs)
    await websocket.send_text(json.dumps({"action": "preview_result", **result}))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    capture_task: Optional[asyncio.Task] = None
    stop_event   = asyncio.Event()
    manual_event = asyncio.Event()

    preview_task: Optional[asyncio.Task] = None
    preview_stop_event = asyncio.Event()

    # Shared camera — kept open across preview↔capture transitions to avoid
    # the 2-4 second DirectShow reopen delay.
    # FPS is measured once on first open and cached.
    shared_cap: Optional[cv2.VideoCapture] = None
    shared_cap_device: Optional[int] = None
    shared_cap_fps: float = 30.0
    shared_cap_w: int = 1920
    shared_cap_h: int = 1080

    loop = asyncio.get_event_loop()

    async def _ensure_cap(message: dict):
        nonlocal shared_cap, shared_cap_device, shared_cap_fps, shared_cap_w, shared_cap_h
        device_raw = message.get("device", 0)
        device = int(device_raw) if str(device_raw).isdigit() else device_raw
        width  = int(message.get("width",  1920))
        height = int(message.get("height", 1080))
        fps    = int(message.get("fps",    30))
        # Reuse if same device and still open
        if shared_cap is not None and shared_cap.isOpened() and shared_cap_device == device:
            return shared_cap
        # Close old cap if different device
        if shared_cap is not None:
            await loop.run_in_executor(_executor, shared_cap.release)
        cap = await loop.run_in_executor(
            _executor, lambda: open_capture(device, width, height, fps)
        )
        if cap is None or not cap.isOpened():
            shared_cap = None
            shared_cap_device = None
            return None
        # Measure fps once here so preview/capture never need to do it
        shared_cap_w, shared_cap_h = await loop.run_in_executor(_executor, read_actual_size, cap)
        shared_cap_fps = await loop.run_in_executor(_executor, read_actual_fps, cap)
        shared_cap = cap
        shared_cap_device = device
        return shared_cap

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON", "raw": raw[:200]}))
                continue

            action = message.get("action")

            if action == "list_devices":
                devices = await loop.run_in_executor(_executor, enumerate_capture_devices)
                await websocket.send_text(json.dumps({
                    "action": "device_list",
                    "devices": devices,
                }))
            elif action == "score_frames":
                await _handle_score_frames(websocket, message)
            elif action == "calibrate":
                await _handle_calibrate(websocket, message)
            elif action == "calibrate_zoom":
                await _handle_calibrate_zoom(websocket, message)
            elif action == "export":
                await _handle_export(websocket, message)
            elif action == "preview_undistort":
                await _handle_preview_undistort(websocket, message)
            elif action == "start_preview":
                if preview_task and not preview_task.done():
                    preview_stop_event.set()
                    await preview_task
                cap = await _ensure_cap(message)
                preview_stop_event.clear()
                # Inject cached measurements so preview doesn't re-measure
                preview_msg = {**message, "_cached_fps": shared_cap_fps,
                               "_cached_w": shared_cap_w, "_cached_h": shared_cap_h}
                preview_task = asyncio.create_task(
                    run_preview(websocket, preview_msg, preview_stop_event, shared_cap=cap)
                )
            elif action == "stop_preview":
                if preview_task and not preview_task.done():
                    preview_stop_event.set()
            elif action == "start_live_capture":
                # Stop preview but keep the camera open — pass shared_cap straight to live capture
                if preview_task and not preview_task.done():
                    preview_stop_event.set()
                    await preview_task
                if capture_task and not capture_task.done():
                    stop_event.set()
                    await capture_task
                cap = await _ensure_cap(message)
                stop_event.clear()
                manual_event.clear()
                # Inject cached measurements so capture doesn't re-measure
                cap_msg = {**message, "_cached_fps": shared_cap_fps,
                           "_cached_w": shared_cap_w, "_cached_h": shared_cap_h}
                capture_task = asyncio.create_task(
                    run_live_capture(websocket, cap_msg, stop_event, manual_event, shared_cap=cap)
                )
            elif action == "stop_live_capture":
                stop_event.set()
            elif action == "manual_capture":
                manual_event.set()
            else:
                await websocket.send_text(json.dumps(message))

    except WebSocketDisconnect:
        if capture_task and not capture_task.done():
            stop_event.set()
    finally:
        # Release shared camera on disconnect
        if shared_cap is not None:
            try:
                await loop.run_in_executor(_executor, shared_cap.release)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
