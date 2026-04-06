"""
Live camera capture with pose-guided frame acquisition for checkerboard calibration.

Protocol
--------
Client → Server:
  {action: "start_live_capture", device: 0, board_cols: 9, board_rows: 6, save_dir: "captures/"}
  {action: "manual_capture"}
  {action: "stop_live_capture"}

Server → Client (streamed):
  {action: "live_frame", frame: "<base64>", found: bool, corners: [[x,y],...],
   quality: str, sharpness: float, coverage: float,
   auto_captured: bool, frame_count: int,
   checklist: [{id, name, hint, satisfied}, ...],
   satisfied_count: int, total: int, complete: bool,
   next_hint: str, next_pose_id: str | None,
   message: str, image_width: int, image_height: int}

  {action: "live_error", error: str}

  {action: "live_capture_stopped", frame_count: int,
   scored_frames: [{path, found, corners, sharpness, coverage, angle, quality, reason,
                    pose_metrics, image_width, image_height}, ...]}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

# Dedicated thread pool — OpenCV releases the GIL, so multiple workers can
# run detection in parallel on multi-core / Hyper-Threaded CPUs.
_executor = ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 4)))

from capture_device import open_capture, read_actual_size, read_actual_fps
from frame_scorer import score_frame_array
from pose_advisor import REQUIRED_POSES, evaluate_checklist, get_required_poses, match_unsatisfied_pose, _pose_matches, compute_border_hint

STREAM_W      = 640    # resolution used for scoring & canvas coordinate space
STREAM_H      = 360
TELE_FALLBACK_W = 960  # second-pass scoring resolution for tele when 640 misses
TELE_FALLBACK_H = 540
HOLD_SECONDS  = 1.5    # how long the board must stay in a matching pose before auto-capture
PREVIEW_FPS   = 15     # target fps for the lightweight preview stream
CAPTURE_FPS   = 8      # target fps for the live-capture loop (scoring is expensive)
PREVIEW_W     = 320    # lower resolution for the idle preview stream (no detection)
PREVIEW_H     = 180    # → 4× less JPEG data, faster encode/transmit/decode
AUTO_CAPTURE_MIN_SHARPNESS = 80   # Laplacian variance floor for auto-capture (Level 1)
COVERAGE_COLS = 10                # coverage grid columns (sent live to frontend)
COVERAGE_ROWS = 6                 # coverage grid rows


def _safe_score_frame_array(image: np.ndarray, checkerboard_size: tuple, fast: bool = False) -> dict:
    """Score a frame without ever raising into the live loop."""
    try:
        return score_frame_array(image, checkerboard_size, fast)
    except Exception as exc:
        h, w = image.shape[:2]
        return {
            "found": False,
            "corners": [],
            "sharpness": 0.0,
            "coverage": 0.0,
            "angle": None,
            "quality": "fail",
            "reason": f"Scoring error: {exc}",
            "image_width": int(w),
            "image_height": int(h),
            "pose_metrics": None,
            "detection_type": None,
            "obj_points": [],
        }


async def run_preview(
    websocket,
    config: dict,
    stop_event: asyncio.Event,
    shared_cap: Optional[cv2.VideoCapture] = None,
) -> None:
    """Lightweight preview stream — no scoring, no saving.  Just streams frames.
    If shared_cap is provided the caller owns it and it will NOT be released here.
    """
    loop = asyncio.get_event_loop()
    owns_cap = shared_cap is None

    if shared_cap is not None and shared_cap.isOpened():
        cap = shared_cap
    else:
        device_raw = config.get("device", 0)
        device     = int(device_raw) if str(device_raw).isdigit() else device_raw
        req_width  = int(config.get("width",  1920))
        req_height = int(config.get("height", 1080))
        req_fps    = int(config.get("fps",    30))
        cap = await loop.run_in_executor(
            None, lambda: open_capture(device, req_width, req_height, req_fps)
        )
        if cap is None or not cap.isOpened():
            await websocket.send_text(json.dumps({
                "action": "preview_error",
                "error": f"Cannot open video device {config.get('device', 0)}",
            }))
            return

    if config.get("_cached_w"):
        actual_w, actual_h = config["_cached_w"], config["_cached_h"]
    else:
        actual_w, actual_h = await loop.run_in_executor(None, read_actual_size, cap)
    actual_fps = config.get("_cached_fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    await websocket.send_text(json.dumps({
        "action":        "preview_started",
        "actual_width":  actual_w,
        "actual_height": actual_h,
        "actual_fps":    round(actual_fps, 3),
    }))

    try:
        while not stop_event.is_set():
            tick = loop.time()
            ret, frame = await loop.run_in_executor(_executor, cap.read)
            if not ret:
                break

            # Preview uses a smaller resolution — purely visual, no detection
            tiny = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            _, buf = cv2.imencode(".jpg", tiny, [cv2.IMWRITE_JPEG_QUALITY, 55])
            frame_b64 = base64.b64encode(buf).decode()

            await websocket.send_text(json.dumps({
                "action": "preview_frame",
                "frame":  frame_b64,
            }))

            elapsed = loop.time() - tick
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=max(0.0, 1.0 / PREVIEW_FPS - elapsed),
                )
            except asyncio.TimeoutError:
                pass
    finally:
        if owns_cap:
            await loop.run_in_executor(None, cap.release)

    await websocket.send_text(json.dumps({"action": "preview_stopped"}))


def _flush_and_grab(cap: cv2.VideoCapture):
    """Drain the camera's internal frame buffer and return the newest frame.

    Capture cards (BMD, AJA, USB) typically queue 1–4 frames.  By calling
    grab() (fast, no decode) several times we skip stale buffered frames and
    retrieve() decodes only the last one — the true "right now" image.
    """
    for _ in range(4):
        if not cap.grab():
            break
    ret, fresh = cap.retrieve()
    return ret, fresh


def _centroid(corners: list) -> Optional[tuple[float, float]]:
    if not corners:
        return None
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _smooth_pose_metrics(prev: Optional[dict], curr: dict, alpha: float = 0.35) -> dict:
    """Exponential smoothing for pose metrics to reduce guidance jitter."""
    if prev is None:
        return dict(curr)

    def ema(key: str, default: float = 0.0) -> float:
        return (1.0 - alpha) * float(prev.get(key, curr.get(key, default))) + alpha * float(curr.get(key, default))

    cx = max(0.0, min(0.999, ema("cx")))
    cy = max(0.0, min(0.999, ema("cy")))
    gx = min(2, int(cx * 3))
    gy = min(2, int(cy * 3))
    region = gy * 3 + gx

    return {
        "region":        region,
        "cx":            round(cx, 3),
        "cy":            round(cy, 3),
        "apparent_size": round(ema("apparent_size"), 4),
        "tilt_score":    round(ema("tilt_score"), 3),
        "border_left":   round(max(0.0, ema("border_left", 0.5)), 3),
        "border_right":  round(max(0.0, ema("border_right", 0.5)), 3),
        "border_top":    round(max(0.0, ema("border_top", 0.5)), 3),
        "border_bottom": round(max(0.0, ema("border_bottom", 0.5)), 3),
    }


def _smooth_points(prev: Optional[np.ndarray], curr: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """EMA smoothing for corner overlays; resets when point count changes."""
    if prev is None or prev.shape != curr.shape:
        return curr
    return (1.0 - alpha) * prev + alpha * curr


def _scale_corners(corners: list, sx: float, sy: float) -> list:
    return [[float(c[0]) * sx, float(c[1]) * sy] for c in corners]


async def run_live_capture(
    websocket,
    config: dict,
    stop_event: asyncio.Event,
    manual_event: asyncio.Event,
    shared_cap: Optional[cv2.VideoCapture] = None,
    target_pose_ref: Optional[list] = None,
) -> None:
    """Live capture with pose-guided auto-capture.
    If shared_cap is provided (already open from preview) it is reused immediately
    with no close/reopen delay.  Caller owns the cap and it will NOT be released here.
    """
    board_cols = int(config.get("board_cols", 9))
    board_rows = int(config.get("board_rows", 6))
    save_dir   = config.get("save_dir", "captures")
    focal_length_mm = float(config.get("focal_length_mm", 0.0) or 0.0)
    manual_only = bool(config.get("manual_only", False))
    checkerboard_size = (board_cols, board_rows)
    fixed_mount    = bool(config.get("fixed_mount", False))
    required_poses = get_required_poses(focal_length_mm or None, fixed_mount=fixed_mount)

    os.makedirs(save_dir, exist_ok=True)

    loop = asyncio.get_event_loop()
    owns_cap = shared_cap is None

    if shared_cap is not None and shared_cap.isOpened():
        cap = shared_cap
    else:
        device_raw = config.get("device", 0)
        device     = int(device_raw) if str(device_raw).isdigit() else device_raw
        req_width  = int(config.get("width",  1920))
        req_height = int(config.get("height", 1080))
        req_fps    = int(config.get("fps",    30))
        cap = await loop.run_in_executor(
            None, lambda: open_capture(device, req_width, req_height, req_fps)
        )
        if cap is None or not cap.isOpened():
            await websocket.send_text(json.dumps({
                "action": "live_error",
                "error": f"Cannot open video device {config.get('device', 0)}",
            }))
            return

    if config.get("_cached_w"):
        actual_w, actual_h = config["_cached_w"], config["_cached_h"]
    else:
        actual_w, actual_h = await loop.run_in_executor(None, read_actual_size, cap)
    actual_fps = config.get("_cached_fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    await websocket.send_text(json.dumps({
        "action":        "live_capture_started",
        "actual_width":  int(actual_w),
        "actual_height": int(actual_h),
        "actual_fps":    round(actual_fps, 3),
    }))

    captured_frames: list[dict] = []

    # Level 3: running coverage grid (native-resolution corners accumulated per frame)
    coverage_grid: list[list[int]] = [[0] * COVERAGE_COLS for _ in range(COVERAGE_ROWS)]

    # Hold-still state — board must match a pose for HOLD_SECONDS before auto-capture
    hold_pose_id:  Optional[str]   = None
    hold_start:    Optional[float] = None
    hold_miss:     int             = 0   # consecutive frames where matching failed

    # Track whether the board was found last iteration to enable the fast
    # detection path (skips slower fallback detectors).
    prev_found = False
    smoothed_pose_metrics: Optional[dict] = None
    smoothed_corners: Optional[np.ndarray] = None

    try:
        while not stop_event.is_set():
            tick = loop.time()

            ret, frame = await loop.run_in_executor(_executor, cap.read)
            if not ret:
                await websocket.send_text(json.dumps({
                    "action": "live_error", "error": "Camera read failed",
                }))
                break

            orig_h, orig_w = frame.shape[:2]

            # Resize to stream resolution FIRST — score on the small image (~9× less
            # work than native 1080p) so the detection loop stays fast enough for
            # the CAPTURE_FPS target.  Corners returned by scoring are already in
            # stream coordinates; no scaling step is needed.
            small = cv2.resize(frame, (STREAM_W, STREAM_H))

            # fast=prev_found skips slow classic-detector fallbacks when the board
            # was detected in the previous frame (the common case while holding still).
            result = await loop.run_in_executor(
                _executor, _safe_score_frame_array, small, checkerboard_size, prev_found
            )

            # Tele fallback: if 640x360 misses, try a second pass on 960x540.
            # This helps coded-target marker detection when downsampled stream
            # loses too much detail at long focal lengths.
            if (not result.get("found")) and focal_length_mm >= 85:
                med = cv2.resize(frame, (TELE_FALLBACK_W, TELE_FALLBACK_H))
                med_result = await loop.run_in_executor(
                    _executor, _safe_score_frame_array, med, checkerboard_size, False
                )
                if med_result.get("found"):
                    med_result["corners"] = _scale_corners(
                        med_result.get("corners", []),
                        STREAM_W / TELE_FALLBACK_W,
                        STREAM_H / TELE_FALLBACK_H,
                    )
                    result = med_result

            prev_found = result["found"]

            pose_metrics_live = result.get("pose_metrics")
            if result["found"] and pose_metrics_live:
                smoothed_pose_metrics = _smooth_pose_metrics(smoothed_pose_metrics, pose_metrics_live)
                pose_metrics_live = smoothed_pose_metrics
            else:
                smoothed_pose_metrics = None

            detection_type_live = result.get("detection_type")
            if result["found"] and result.get("corners"):
                corner_arr = np.array(result["corners"], dtype=np.float32).reshape(-1, 2)
                # Only smooth ordered checkerboard corners. ArUco/grid point order
                # is not stable frame-to-frame, so index-wise EMA causes twitching.
                if detection_type_live == "checkerboard":
                    smoothed_corners = _smooth_points(smoothed_corners, corner_arr)
                    stream_corners = smoothed_corners.tolist()
                else:
                    smoothed_corners = None
                    stream_corners = corner_arr.tolist()
            else:
                smoothed_corners = None
                stream_corners = []

            # Use smoothed metrics for matching/display; raw metrics remain stored
            # on captured frames from native_result.
            result["pose_metrics"] = pose_metrics_live

            # Pose checklist based on all captured frames so far
            checklist_info = evaluate_checklist(captured_frames, required_poses)

            # ── Hold-still auto-capture logic ──────────────────────────────
            force = manual_event.is_set()
            if force:
                manual_event.clear()

            auto_captured  = False
            hold_progress  = 0.0   # 0–1 sent to frontend for the hold ring
            matching_pose  = None  # which unsatisfied pose the frame currently matches

            pinned_pose_id = target_pose_ref[0] if target_pose_ref else None

            if result["found"] and pose_metrics_live:
                if checklist_info["complete"]:
                    # After required poses are complete, keep auto-capturing stable
                    # board views so operators can gather denser frame sets.
                    matching_pose = "bonus_any"
                elif pinned_pose_id:
                    # User has selected a specific pose — only match against that one
                    satisfied_ids = {p["id"] for p in checklist_info["checklist"] if p["satisfied"]}
                    pinned_def = next((p for p in required_poses if p["id"] == pinned_pose_id), None)
                    if pinned_def and pinned_pose_id not in satisfied_ids and _pose_matches(pose_metrics_live, pinned_def):
                        matching_pose = pinned_pose_id
                else:
                    matching_pose = await loop.run_in_executor(
                        _executor, match_unsatisfied_pose,
                        pose_metrics_live, captured_frames, required_poses,
                    )

            if matching_pose:
                hold_miss = 0
                now = loop.time()
                if hold_pose_id != matching_pose:
                    # New pose matched — reset hold timer
                    hold_pose_id = matching_pose
                    hold_start   = now
                hold_progress = min(1.0, (now - hold_start) / HOLD_SECONDS)
            else:
                # Allow a short grace period before resetting hold timer
                hold_miss += 1
                if hold_miss >= 4:
                    hold_pose_id = None
                    hold_start   = None
                    hold_miss    = 0

            # Level 1: gate auto-capture on sharpness; manual force always allowed
            auto_sharpness_ok = result["sharpness"] >= AUTO_CAPTURE_MIN_SHARPNESS
            do_capture = force or ((not manual_only) and hold_progress >= 1.0 and matching_pose is not None and auto_sharpness_ok)

            if do_capture:
                # Use nanosecond timestamp to avoid collisions when two captures
                # happen within the same millisecond (e.g. rapid manual triggers)
                ts = time.time_ns()
                save_path = os.path.join(save_dir, f"frame_{ts}.jpg")
                # Flush the camera buffer and grab the freshest raw frame directly
                # from the I/O card.  The scoring loop runs at CAPTURE_FPS (8fps)
                # but the camera pushes frames at 30fps, so the buffer may hold
                # several newer frames.  _flush_and_grab() drains those and
                # retrieves the most current image — no JPEG decode in between.
                ret_fresh, fresh_frame = await loop.run_in_executor(
                    _executor, _flush_and_grab, cap
                )
                capture_frame = fresh_frame if ret_fresh else frame
                capture_orig_h, capture_orig_w = capture_frame.shape[:2]

                # Score the native-resolution frame for full-precision corners.
                # This runs only once per HOLD_SECONDS — does not affect preview FPS.
                # NOTE: We do NOT re-verify pose metrics after the flush.  The hold
                # timer (HOLD_SECONDS) is the quality gate; re-testing on the
                # freshly-grabbed image rejects valid frames at telephoto because
                # tiny FOV + small board appearance cause metric fluctuations even
                # with a perfectly stationary board.
                native_result = await loop.run_in_executor(
                    _executor, _safe_score_frame_array, capture_frame, checkerboard_size
                )
                # If the freshest frame has no board (camera lag edge-case),
                # fall back to the current loop frame rather than discarding.
                if not native_result.get("found"):
                    fallback_result = await loop.run_in_executor(
                        _executor, _safe_score_frame_array, frame, checkerboard_size
                    )
                    if fallback_result.get("found"):
                        native_result = fallback_result
                        capture_frame = frame
                        capture_orig_h, capture_orig_w = capture_frame.shape[:2]

                await loop.run_in_executor(
                    _executor,
                    lambda p=save_path, f=capture_frame: cv2.imwrite(p, f, [cv2.IMWRITE_JPEG_QUALITY, 95]),
                )
                # For manual capture, keep the frame even if no chart is found.
                if force and not native_result.get("found"):
                    native_result["reason"] = native_result.get("reason") or "Manual capture without detected chart"
                # Store native-resolution corners + dimensions for calibration.
                # stream_corners are only used in the live_frame overlay below.
                captured_frames.append({
                    **native_result,
                    "path": save_path,
                    "image_width":  capture_orig_w,
                    "image_height": capture_orig_h,
                    "captured_pose_id": matching_pose if matching_pose != "bonus_any" else None,
                })
                # Level 3: update running coverage grid from native-resolution corners
                if capture_orig_w > 0 and capture_orig_h > 0:
                    for cx_n, cy_n in native_result.get("corners", []):
                        col = min(COVERAGE_COLS - 1, int(cx_n / capture_orig_w * COVERAGE_COLS))
                        row = min(COVERAGE_ROWS - 1, int(cy_n / capture_orig_h * COVERAGE_ROWS))
                        coverage_grid[row][col] += 1

                print(f"[live_capture] captured frame: quality={native_result['quality']}, sharpness={native_result['sharpness']}, coverage={native_result['coverage']:.2%}, pose={matching_pose} (flushed={ret_fresh})")
                # Reset hold so next pose can start fresh
                hold_pose_id = None
                hold_start   = None
                hold_progress = 0.0
                auto_captured = True

                # Re-evaluate after capture
                checklist_info = evaluate_checklist(captured_frames, required_poses)

            # Encode stream frame (already resized to STREAM_W×STREAM_H above)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 65])
            frame_b64 = base64.b64encode(buf).decode()

            # Visual guidance: target zone + per-requirement match status
            # When user has pinned a pose, show guidance for that pose instead of
            # the auto-selected next unsatisfied pose.
            next_pose_reqs = None
            match_status   = None
            satisfied_ids_for_display = {p["id"] for p in checklist_info["checklist"] if p["satisfied"]}
            display_pose_id = (
                pinned_pose_id
                if pinned_pose_id and pinned_pose_id not in satisfied_ids_for_display
                else checklist_info["next_pose_id"]
            )
            display_next_hint = checklist_info["next_hint"]
            if pinned_pose_id and pinned_pose_id not in satisfied_ids_for_display and not checklist_info["complete"]:
                pinned_def_display = next((p for p in required_poses if p["id"] == pinned_pose_id), None)
                if pinned_def_display:
                    display_next_hint = pinned_def_display["hint"]
            if not checklist_info["complete"] and display_pose_id:
                next_def = next(
                    (p for p in required_poses if p["id"] == display_pose_id), None
                )
                if next_def:
                    next_pose_reqs = {
                        "regions":  next_def["region"],
                        "tilt_min": next_def["tilt_min"],
                        "tilt_max": next_def["tilt_max"],
                        "size_min": next_def["size_min"],
                    }
                    if result["found"] and pose_metrics_live:
                        m = pose_metrics_live
                        tilt_hint = "good"
                        if m["tilt_score"] < next_def["tilt_min"]:
                            tilt_hint = "Tilt more — rotate the board left/right or tip it toward you"
                        elif m["tilt_score"] > next_def["tilt_max"]:
                            tilt_hint = "Tilt less — face the board more directly at the camera"

                        size_hint = "good"
                        if m["apparent_size"] > 0.85:
                            size_hint = "Move back — chart is too large in frame"
                        elif m["apparent_size"] < next_def["size_min"]:
                            size_hint = "Move closer — chart is too small"

                        position_ok = (next_def["region"] is None or m["region"] in next_def["region"])
                        position_hint = None
                        if not position_ok and next_def["region"] is not None:
                            target_cols = [r % 3 for r in next_def["region"]]
                            target_rows = [r // 3 for r in next_def["region"]]
                            avg_tc = sum(target_cols) / len(target_cols)
                            avg_tr = sum(target_rows) / len(target_rows)
                            curr_col = m["region"] % 3
                            curr_row = m["region"] // 3
                            dirs = []
                            if curr_col < avg_tc - 0.4:
                                dirs.append("right")
                            elif curr_col > avg_tc + 0.4:
                                dirs.append("left")
                            if curr_row < avg_tr - 0.4:
                                dirs.append("down")
                            elif curr_row > avg_tr + 0.4:
                                dirs.append("up")
                            if dirs:
                                position_hint = "Move " + " & ".join(dirs)

                        # Level 2: border proximity hint
                        border_hint = compute_border_hint(m, next_def)

                        # Level 1: sharpness feedback
                        sharp = result["sharpness"]
                        sharpness_ok = sharp >= AUTO_CAPTURE_MIN_SHARPNESS
                        if sharp >= 200:
                            sharpness_hint = "Sharp"
                        elif sharp >= AUTO_CAPTURE_MIN_SHARPNESS:
                            sharpness_hint = "good"
                        elif sharp >= 30:
                            sharpness_hint = "Hold very still — slightly blurry"
                        else:
                            sharpness_hint = "Very blurry — hold still or check focus"

                        match_status = {
                            "position_ok":    position_ok,
                            "position_hint":  position_hint,
                            "tilt_ok":        next_def["tilt_min"] <= m["tilt_score"] <= next_def["tilt_max"],
                            "size_ok":        m["apparent_size"] >= next_def["size_min"],
                            "tilt_score":     m["tilt_score"],
                            "tilt_hint":      tilt_hint,
                            "size_score":     m["apparent_size"],
                            "size_hint":      size_hint,
                            "border_hint":    border_hint,
                            "sharpness_ok":   sharpness_ok,
                            "sharpness_score": round(sharp, 1),
                            "sharpness_hint": sharpness_hint,
                        }

            msg_text = "Board detected" if result["found"] else "No board detected"
            if result["found"]:
                msg_text += f" — {result['quality']}"

            covered_cells = sum(1 for row in coverage_grid for c in row if c > 0)
            frame_coverage_pct = round(covered_cells / (COVERAGE_COLS * COVERAGE_ROWS) * 100, 1)

            await websocket.send_text(json.dumps({
                "action":               "live_frame",
                "frame":                frame_b64,
                "found":                result["found"],
                "detection_type":       result.get("detection_type"),
                "corners":              stream_corners,
                "quality":              result["quality"],
                "sharpness":            result["sharpness"],
                "coverage":             result["coverage"],
                "auto_captured":        auto_captured,
                "frame_count":          len(captured_frames),
                "checklist":            checklist_info["checklist"],
                "satisfied_count":      checklist_info["satisfied_count"],
                "total":                checklist_info["total"],
                "complete":             checklist_info["complete"],
                "next_hint":            display_next_hint,
                "next_pose_id":         display_pose_id,
                "matching_pose_id":     matching_pose,
                "hold_progress":        round(hold_progress, 3),
                "message":              msg_text,
                "image_width":          STREAM_W,
                "image_height":         STREAM_H,
                "next_pose_reqs":       next_pose_reqs,
                "match_status":         match_status,
                "coverage_grid":        coverage_grid,
                "frame_coverage_pct":   frame_coverage_pct,
            }))

            # Pace to CAPTURE_FPS; allow early exit on stop
            elapsed = loop.time() - tick
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=max(0.0, 1.0 / CAPTURE_FPS - elapsed),
                )
            except asyncio.TimeoutError:
                pass

    finally:
        if owns_cap:
            await loop.run_in_executor(None, cap.release)

    # Summarize quality breakdown for diagnostics
    good_count = sum(1 for f in captured_frames if f["quality"] == "good")
    warn_count = sum(1 for f in captured_frames if f["quality"] == "warn")
    fail_count = sum(1 for f in captured_frames if f["quality"] == "fail")
    usable_count = good_count + warn_count
    print(f"[live_capture] Stopped: {len(captured_frames)} frames total | {good_count} good + {warn_count} warn = {usable_count} usable | {fail_count} rejected")

    await websocket.send_text(json.dumps({
        "action":        "live_capture_stopped",
        "frame_count":   len(captured_frames),
        "scored_frames": captured_frames,
    }))
