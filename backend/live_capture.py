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
from typing import Optional

import cv2
import numpy as np

from capture_device import open_capture, read_actual_size
from frame_scorer import score_frame_array
from pose_advisor import evaluate_checklist, match_unsatisfied_pose

STREAM_W     = 640
STREAM_H     = 360
HOLD_SECONDS = 0.8   # how long the board must stay in a matching pose before auto-capture


def _centroid(corners: list) -> Optional[tuple[float, float]]:
    if not corners:
        return None
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return sum(xs) / len(xs), sum(ys) / len(ys)


async def run_live_capture(
    websocket,
    config: dict,
    stop_event: asyncio.Event,
    manual_event: asyncio.Event,
) -> None:
    device_raw = config.get("device", 0)
    device     = int(device_raw) if str(device_raw).isdigit() else device_raw
    board_cols = int(config.get("board_cols", 9))
    board_rows = int(config.get("board_rows", 6))
    save_dir   = config.get("save_dir", "captures")
    req_width  = int(config.get("width",  1920))
    req_height = int(config.get("height", 1080))
    req_fps    = int(config.get("fps",    30))
    checkerboard_size = (board_cols, board_rows)

    os.makedirs(save_dir, exist_ok=True)

    loop = asyncio.get_event_loop()

    cap: Optional[cv2.VideoCapture] = await loop.run_in_executor(
        None, lambda: open_capture(device, req_width, req_height, req_fps)
    )
    if cap is None or not cap.isOpened():
        await websocket.send_text(json.dumps({
            "action": "live_error",
            "error": f"Cannot open video device {device}",
        }))
        return

    # Frames stored at native resolution for calibration; path + pose_metrics included
    # Read back actual resolution (SDI cards lock to signal, ignore requests)
    actual_w, actual_h = await loop.run_in_executor(None, read_actual_size, cap)

    # Send device info so the UI can update its image_size for calibration
    await websocket.send_text(json.dumps({
        "action":        "live_capture_started",
        "actual_width":  actual_w,
        "actual_height": actual_h,
    }))

    captured_frames: list[dict] = []

    # Hold-still state — board must match a pose for HOLD_SECONDS before auto-capture
    hold_pose_id:  Optional[str]   = None
    hold_start:    Optional[float] = None

    try:
        while not stop_event.is_set():
            tick = loop.time()

            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                await websocket.send_text(json.dumps({
                    "action": "live_error", "error": "Camera read failed",
                }))
                break

            orig_h, orig_w = frame.shape[:2]

            # Score at native resolution (includes pose_metrics)
            result = await loop.run_in_executor(
                None, score_frame_array, frame, checkerboard_size
            )

            # Scale corners to stream resolution for display
            sx = STREAM_W / orig_w
            sy = STREAM_H / orig_h
            stream_corners = [
                [c[0] * sx, c[1] * sy] for c in result["corners"]
            ]

            # Pose checklist based on all captured frames so far
            checklist_info = evaluate_checklist(captured_frames)

            # ── Hold-still auto-capture logic ──────────────────────────────
            force = manual_event.is_set()
            if force:
                manual_event.clear()

            auto_captured  = False
            hold_progress  = 0.0   # 0–1 sent to frontend for the hold ring
            matching_pose  = None  # which unsatisfied pose the frame currently matches

            if not checklist_info["complete"]:
                if result["found"] and result["quality"] != "fail" and result.get("pose_metrics"):
                    matching_pose = await loop.run_in_executor(
                        None, match_unsatisfied_pose,
                        result["pose_metrics"], captured_frames,
                    )

                if matching_pose:
                    now = loop.time()
                    if hold_pose_id != matching_pose:
                        # New pose matched — reset hold timer
                        hold_pose_id = matching_pose
                        hold_start   = now
                    hold_progress = min(1.0, (now - hold_start) / HOLD_SECONDS)
                else:
                    # Board not matching anything — reset
                    hold_pose_id = None
                    hold_start   = None

            do_capture = force or (hold_progress >= 1.0 and matching_pose is not None)

            if do_capture:
                ts = int(time.time() * 1000)
                save_path = os.path.join(save_dir, f"frame_{ts}.jpg")
                await loop.run_in_executor(
                    None,
                    lambda p=save_path: cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]),
                )
                captured_frames.append({
                    **result,
                    "corners":      stream_corners,
                    "path":         save_path,
                    "image_width":  STREAM_W,
                    "image_height": STREAM_H,
                })
                # Reset hold so next pose can start fresh
                hold_pose_id = None
                hold_start   = None
                hold_progress = 0.0
                auto_captured = True

                # Re-evaluate after capture
                checklist_info = evaluate_checklist(captured_frames)

            # Encode stream frame
            small = cv2.resize(frame, (STREAM_W, STREAM_H))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 65])
            frame_b64 = base64.b64encode(buf).decode()

            msg_text = "Board detected" if result["found"] else "No board detected"
            if result["found"]:
                msg_text += f" — {result['quality']}"

            await websocket.send_text(json.dumps({
                "action":           "live_frame",
                "frame":            frame_b64,
                "found":            result["found"],
                "corners":          stream_corners,
                "quality":          result["quality"],
                "sharpness":        result["sharpness"],
                "coverage":         result["coverage"],
                "auto_captured":    auto_captured,
                "frame_count":      len(captured_frames),
                "checklist":        checklist_info["checklist"],
                "satisfied_count":  checklist_info["satisfied_count"],
                "total":            checklist_info["total"],
                "complete":         checklist_info["complete"],
                "next_hint":        checklist_info["next_hint"],
                "next_pose_id":     checklist_info["next_pose_id"],
                "matching_pose_id": matching_pose,
                "hold_progress":    round(hold_progress, 3),
                "message":          msg_text,
                "image_width":      STREAM_W,
                "image_height":     STREAM_H,
            }))

            # Pace to ~5 fps; allow early exit on stop
            elapsed = loop.time() - tick
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=max(0.0, 0.2 - elapsed))
            except asyncio.TimeoutError:
                pass

    finally:
        await loop.run_in_executor(None, cap.release)

    await websocket.send_text(json.dumps({
        "action":        "live_capture_stopped",
        "frame_count":   len(captured_frames),
        "scored_frames": captured_frames,
    }))
