"""
Pose advisor for calibration frame acquisition.

Rather than capturing 25 random frames, we guide the user to capture
10 specific poses that together fully constrain all distortion parameters.

The 10 required poses sample:
  - Centre flat          → constrains principal point + radial baseline
  - Centre tilted ×2     → constrains K2/K3 (higher-order radial terms)
  - 4 corner positions   → constrains field-edge distortion
  - Close-up             → constrains focal length + principal point accuracy
  - Strong tilt          → separates radial from tangential (P1/P2)
  - Top + Bottom edges   → constrains vertical distortion independently

Mathematical basis: each pose adds independent rows to the design matrix
of the calibration system.  These 10 poses maximise the rank / condition
number of that matrix with minimal shots.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Required pose definitions
# ---------------------------------------------------------------------------

REQUIRED_POSES: list[dict] = [
    {
        "id":        "center_flat",
        "name":      "Centre — flat",
        "hint":      "Hold the board in the middle of the frame, facing the camera straight on",
        "region":    [4],           # centre cell of 3×3 grid
        "tilt_max":  0.15,
        "tilt_min":  0.00,
        "size_min":  0.15,
    },
    {
        "id":        "center_tilted_h",
        "name":      "Centre — tilted left/right",
        "hint":      "Keep the board centred but rotate it left or right ~30°",
        "region":    [3, 4, 5],
        "tilt_max":  1.00,
        "tilt_min":  0.25,
        "size_min":  0.10,
    },
    {
        "id":        "center_tilted_v",
        "name":      "Centre — tilted up/down",
        "hint":      "Keep the board centred but tilt the top toward or away from you ~20°",
        "region":    [1, 4, 7],
        "tilt_max":  1.00,
        "tilt_min":  0.20,
        "size_min":  0.10,
    },
    {
        "id":        "corner_tl",
        "name":      "Top-left corner",
        "hint":      "Move the board to the top-left corner of the frame",
        "region":    [0],
        "tilt_max":  1.00,
        "tilt_min":  0.00,
        "size_min":  0.06,
    },
    {
        "id":        "corner_tr",
        "name":      "Top-right corner",
        "hint":      "Move the board to the top-right corner of the frame",
        "region":    [2],
        "tilt_max":  1.00,
        "tilt_min":  0.00,
        "size_min":  0.06,
    },
    {
        "id":        "corner_bl",
        "name":      "Bottom-left corner",
        "hint":      "Move the board to the bottom-left corner of the frame",
        "region":    [6],
        "tilt_max":  1.00,
        "tilt_min":  0.00,
        "size_min":  0.06,
    },
    {
        "id":        "corner_br",
        "name":      "Bottom-right corner",
        "hint":      "Move the board to the bottom-right corner of the frame",
        "region":    [8],
        "tilt_max":  1.00,
        "tilt_min":  0.00,
        "size_min":  0.06,
    },
    {
        "id":        "close_up",
        "name":      "Close-up (large in frame)",
        "hint":      "Move the board closer so it fills about a quarter of the frame",
        "region":    None,          # any region is fine
        "tilt_max":  0.30,
        "tilt_min":  0.00,
        "size_min":  0.20,
    },
    {
        "id":        "strong_tilt",
        "name":      "Strong tilt (any axis)",
        "hint":      "Tilt the board toward or away from you — tip the top edge forward or lean it sideways so it looks clearly non-flat",
        "region":    None,
        "tilt_max":  1.00,
        "tilt_min":  0.20,
        "size_min":  0.05,
    },
    {
        "id":        "top_or_bottom",
        "name":      "Top or bottom edge",
        "hint":      "Place the board along the top or bottom edge of the frame",
        "region":    [1, 7],
        "tilt_max":  1.00,
        "tilt_min":  0.00,
        "size_min":  0.06,
    },
]


# ---------------------------------------------------------------------------
# Pose metric extraction (no camera matrix required)
# ---------------------------------------------------------------------------

def compute_pose_metrics(
    corners: list,
    checkerboard_size: tuple,  # (cols, rows)
    image_size: tuple,          # (width, height)
) -> dict:
    """
    Estimate pose region, tilt, and apparent size from raw corner positions.
    All values are normalised so they work before camera calibration.
    """
    cols, rows = checkerboard_size
    pts = np.array(corners, dtype=np.float64).reshape(-1, 2)

    iw, ih = image_size

    # Board centre (normalised)
    cx = float(pts[:, 0].mean()) / iw
    cy = float(pts[:, 1].mean()) / ih

    # 3×3 grid region (0=top-left … 8=bottom-right)
    gx = min(2, int(cx * 3))
    gy = min(2, int(cy * 3))
    region = gy * 3 + gx

    # Apparent area as fraction of image
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    apparent_size = float((x_max - x_min) * (y_max - y_min)) / (iw * ih)

    # Tilt score via perspective foreshortening of the projected grid
    grid = pts.reshape(rows, cols, 2)

    top_w   = float(np.linalg.norm(grid[0, -1]  - grid[0,  0]))
    bot_w   = float(np.linalg.norm(grid[-1, -1] - grid[-1, 0]))
    left_h  = float(np.linalg.norm(grid[-1, 0]  - grid[0,  0]))
    right_h = float(np.linalg.norm(grid[-1, -1] - grid[0, -1]))

    w_ratio = min(top_w, bot_w)  / max(top_w, bot_w,  1e-6)
    h_ratio = min(left_h, right_h) / max(left_h, right_h, 1e-6)
    tilt_score = float(1.0 - min(w_ratio, h_ratio))

    return {
        "region":        region,
        "cx":            round(cx, 3),
        "cy":            round(cy, 3),
        "apparent_size": round(apparent_size, 4),
        "tilt_score":    round(tilt_score, 3),
    }


# ---------------------------------------------------------------------------
# Checklist management
# ---------------------------------------------------------------------------

def _pose_matches(metrics: dict, pose_def: dict) -> bool:
    if metrics["tilt_score"] < pose_def["tilt_min"]:
        return False
    if metrics["tilt_score"] > pose_def["tilt_max"]:
        return False
    if metrics["apparent_size"] < pose_def["size_min"]:
        return False
    if pose_def["region"] is not None and metrics["region"] not in pose_def["region"]:
        return False
    return True


def match_unsatisfied_pose(metrics: dict, captured_frames: list) -> str | None:
    """
    Return the id of the first unsatisfied pose that the current frame satisfies,
    or None if the frame doesn't match any outstanding pose.
    Used by the auto-capture hold logic.
    """
    satisfied_ids: set[str] = set()
    for frame in captured_frames:
        m = frame.get("pose_metrics")
        if not m:
            continue
        for pose_def in REQUIRED_POSES:
            if pose_def["id"] not in satisfied_ids and _pose_matches(m, pose_def):
                satisfied_ids.add(pose_def["id"])

    for pose_def in REQUIRED_POSES:
        if pose_def["id"] not in satisfied_ids and _pose_matches(metrics, pose_def):
            return pose_def["id"]
    return None


def evaluate_checklist(captured_frames: list) -> dict:
    """
    Given a list of captured frames (each with a 'pose_metrics' key),
    return which poses are satisfied and what to do next.

    Returns:
        {
          checklist: [{id, name, hint, satisfied: bool}, ...],
          satisfied_count: int,
          total: int,
          complete: bool,
          next_hint: str,
          next_pose_id: str | None,
        }
    """
    satisfied_ids: set[str] = set()

    for frame in captured_frames:
        metrics = frame.get("pose_metrics")
        if not metrics:
            continue
        for pose_def in REQUIRED_POSES:
            if pose_def["id"] not in satisfied_ids and _pose_matches(metrics, pose_def):
                satisfied_ids.add(pose_def["id"])

    checklist = [
        {
            "id":        p["id"],
            "name":      p["name"],
            "hint":      p["hint"],
            "satisfied": p["id"] in satisfied_ids,
        }
        for p in REQUIRED_POSES
    ]

    remaining = [p for p in REQUIRED_POSES if p["id"] not in satisfied_ids]
    complete = len(remaining) == 0

    next_hint = "All poses captured! Ready to calibrate." if complete else remaining[0]["hint"]
    next_pose_id = None if complete else remaining[0]["id"]

    return {
        "checklist":        checklist,
        "satisfied_count":  len(satisfied_ids),
        "total":            len(REQUIRED_POSES),
        "complete":         complete,
        "next_hint":        next_hint,
        "next_pose_id":     next_pose_id,
    }


# ---------------------------------------------------------------------------
# Convergence check (run after each new frame)
# ---------------------------------------------------------------------------

def should_stop_early(rms_history: list[float]) -> tuple[bool, str]:
    """
    Return (should_stop, reason) based on RMS convergence.
    Called after each progressive calibration run.
    """
    if len(rms_history) < 3:
        return False, ""

    current = rms_history[-1]
    previous = rms_history[-2]

    improvement = abs(previous - current) / max(previous, 1e-6)

    if current < 0.3 and improvement < 0.03:
        return True, f"Excellent — RMS {current:.3f} px converged (< 0.3 px, < 3% change)"
    if current < 0.5 and improvement < 0.05:
        return True, f"Good — RMS {current:.3f} px converged (< 0.5 px, < 5% change)"
    if current < 1.0 and len(rms_history) >= 5 and improvement < 0.02:
        return True, f"Marginal but stable — RMS {current:.3f} px (< 1.0 px, < 2% change)"

    return False, ""
