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

import cv2
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
        "id":        "left_edge",
        "name":      "Left edge",
        "hint":      "Move the board to the centre-left of the frame (halfway up, left side)",
        "region":    [3],
        "tilt_max":  1.00,
        "tilt_min":  0.00,
        "size_min":  0.06,
    },
    {
        "id":        "right_edge",
        "name":      "Right edge",
        "hint":      "Move the board to the centre-right of the frame (halfway up, right side)",
        "region":    [5],
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
FIXED_MOUNT_REQUIRED_POSES: list[dict] = [
    {
        "id":       "center_flat",
        "name":     "Centre — flat",
        "hint":     "Aim the camera directly at the chart, lens facing it straight on",
        "region":   [4],
        "tilt_max": 0.15,
        "tilt_min": 0.00,
        "size_min": 0.05,
    },
    {
        "id":       "center_tilted_h",
        "name":     "Pan left or right",
        "hint":     "Pan the camera left or right so the chart shifts horizontally but stays near centre",
        "region":   [3, 4, 5],
        "tilt_max": 1.00,
        "tilt_min": 0.20,
        "size_min": 0.04,
    },
    {
        "id":       "center_tilted_v",
        "name":     "Tilt up or down",
        "hint":     "Tilt the camera up or down so the chart shifts vertically but stays near centre",
        "region":   [1, 4, 7],
        "tilt_max": 1.00,
        "tilt_min": 0.15,
        "size_min": 0.04,
    },
    {
        "id":       "corner_tl",
        "name":     "Pan to top-left",
        "hint":     "Pan and tilt the camera until the chart appears in the top-left of the frame",
        "region":   [0],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
    {
        "id":       "corner_tr",
        "name":     "Pan to top-right",
        "hint":     "Pan and tilt the camera until the chart appears in the top-right of the frame",
        "region":   [2],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
    {
        "id":       "corner_bl",
        "name":     "Pan to bottom-left",
        "hint":     "Pan and tilt the camera until the chart appears in the bottom-left of the frame",
        "region":   [6],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
    {
        "id":       "corner_br",
        "name":     "Pan to bottom-right",
        "hint":     "Pan and tilt the camera until the chart appears in the bottom-right of the frame",
        "region":   [8],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
    {
        "id":       "left_edge",
        "name":     "Pan to left edge",
        "hint":     "Pan the camera so the chart appears in the centre-left of the frame",
        "region":   [3],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
    {
        "id":       "right_edge",
        "name":     "Pan to right edge",
        "hint":     "Pan the camera so the chart appears in the centre-right of the frame",
        "region":   [5],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
    {
        "id":       "offcenter_tilt",
        "name":     "Off-centre with angle",
        "hint":     "Pan camera to an edge so the chart is off-centre AND shows visible foreshortening",
        "region":   [0, 1, 2, 3, 5, 6, 7, 8],
        "tilt_max": 1.00,
        "tilt_min": 0.20,
        "size_min": 0.04,
    },
    {
        "id":       "strong_tilt",
        "name":     "Strong pan/tilt angle",
        "hint":     "Pan or tilt the camera to a steep angle — strong foreshortening visible on the chart",
        "region":   None,
        "tilt_max": 1.00,
        "tilt_min": 0.20,
        "size_min": 0.04,
    },
    {
        "id":       "top_or_bottom",
        "name":     "Top or bottom edge",
        "hint":     "Tilt the camera up or down to place the chart along the top or bottom of the frame",
        "region":   [1, 7],
        "tilt_max": 1.00,
        "tilt_min": 0.00,
        "size_min": 0.04,
    },
]


def get_required_poses(focal_length_mm: float | None = None, fixed_mount: bool = False) -> list[dict]:
    """Return pose definitions adjusted for the current focal length and mount mode.

    fixed_mount=True  — chart is stationary on a stand; camera pans/tilts/zooms only.
    fixed_mount=False — classic handheld mode: the chart is moved to each pose.

    Wide focal lengths produce less apparent foreshortening for the same
    physical board rotation, so the raw tilt score is lower. Relax the wide
    angle thresholds so 28-35 mm shots can still satisfy the intended poses.
    """
    if fixed_mount:
        poses = [dict(p) for p in FIXED_MOUNT_REQUIRED_POSES]
        # In fixed-mount mode, wide FL produces very small board apparent size
        # and less foreshortening — relax tilt_min thresholds.
        if focal_length_mm is not None:
            if focal_length_mm <= 35:
                fm_adjustments = {
                    "center_tilted_h": {"tilt_min": 0.13},
                    "center_tilted_v": {"tilt_min": 0.11},
                    "offcenter_tilt":  {"tilt_min": 0.13},
                    "strong_tilt":     {"tilt_min": 0.13},
                    "left_edge":       {"region": [0, 3, 6]},
                    "right_edge":      {"region": [2, 5, 8]},
                }
            elif focal_length_mm < 50:
                fm_adjustments = {
                    "center_tilted_h": {"tilt_min": 0.15},
                    "center_tilted_v": {"tilt_min": 0.13},
                    "offcenter_tilt":  {"tilt_min": 0.15},
                    "strong_tilt":     {"tilt_min": 0.15},
                    "left_edge":       {"region": [0, 3, 6]},
                    "right_edge":      {"region": [2, 5, 8]},
                }
            elif focal_length_mm < 85:
                fm_adjustments = {
                    "corner_tl":  {"region": [0, 1, 3],    "size_min": 0.03},
                    "corner_tr":  {"region": [1, 2, 5],    "size_min": 0.03},
                    "corner_bl":  {"region": [3, 6, 7],    "size_min": 0.03},
                    "corner_br":  {"region": [5, 7, 8],    "size_min": 0.03},
                    "left_edge":  {"region": [0, 3, 6],    "size_min": 0.03},
                    "right_edge": {"region": [2, 5, 8],    "size_min": 0.03},
                }
            elif focal_length_mm < 135:
                fm_adjustments = {
                    "center_tilted_h": {"tilt_min": 0.08, "region": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                    "center_tilted_v": {"tilt_min": 0.07, "region": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                    "offcenter_tilt":  {"tilt_min": 0.08},
                    "strong_tilt":     {"tilt_min": 0.07},
                    "corner_tl":  {"region": [0, 1, 3, 4], "size_min": 0.025},
                    "corner_tr":  {"region": [1, 2, 4, 5], "size_min": 0.025},
                    "corner_bl":  {"region": [3, 4, 6, 7], "size_min": 0.025},
                    "corner_br":  {"region": [4, 5, 7, 8], "size_min": 0.025},
                    "left_edge":  {"region": [0, 1, 3, 4], "size_min": 0.025},
                    "right_edge": {"region": [1, 2, 4, 5], "size_min": 0.025},
                }
            else:
                fm_adjustments = {
                    "center_tilted_h": {"tilt_min": 0.06, "region": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                    "center_tilted_v": {"tilt_min": 0.05, "region": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                    "offcenter_tilt":  {"tilt_min": 0.06},
                    "strong_tilt":     {"tilt_min": 0.05},
                    "corner_tl":  {"region": [0, 1, 3, 4], "size_min": 0.02},
                    "corner_tr":  {"region": [1, 2, 4, 5], "size_min": 0.02},
                    "corner_bl":  {"region": [3, 4, 6, 7], "size_min": 0.02},
                    "corner_br":  {"region": [4, 5, 7, 8], "size_min": 0.02},
                    "left_edge":  {"region": [0, 1, 3, 4, 6, 7], "size_min": 0.02},
                    "right_edge": {"region": [1, 2, 4, 5, 7, 8], "size_min": 0.02},
                }
            for pose in poses:
                update = fm_adjustments.get(pose["id"])
                if update:
                    pose.update(update)
        return poses

    poses = [dict(p) for p in REQUIRED_POSES]
    if focal_length_mm is None:
        return poses

    if focal_length_mm <= 35:
        adjustments = {
            "center_tilted_h": {"tilt_min": 0.16, "size_min": 0.08},
            "center_tilted_v": {"tilt_min": 0.14, "size_min": 0.08},
            "close_up":        {"size_min": 0.16},
            "strong_tilt":     {"tilt_min": 0.14, "size_min": 0.04},
            "left_edge":       {"region": [0, 3, 6], "size_min": 0.05},
            "right_edge":      {"region": [2, 5, 8], "size_min": 0.05},
        }
    elif focal_length_mm < 50:
        adjustments = {
            "center_tilted_h": {"tilt_min": 0.20, "size_min": 0.09},
            "center_tilted_v": {"tilt_min": 0.17, "size_min": 0.09},
            "close_up":        {"size_min": 0.18},
            "strong_tilt":     {"tilt_min": 0.16, "size_min": 0.045},
            "left_edge":       {"region": [0, 3, 6], "size_min": 0.05},
            "right_edge":      {"region": [2, 5, 8], "size_min": 0.05},
        }
    elif focal_length_mm < 85:
        adjustments = {
            "center_tilted_h": {"tilt_min": 0.18},
            "center_tilted_v": {"tilt_min": 0.15},
            "strong_tilt":     {"tilt_min": 0.15},
            # 50-85mm: start relaxing corners — board at edge is harder to achieve
            "corner_tl":       {"region": [0, 1, 3],    "size_min": 0.05},
            "corner_tr":       {"region": [1, 2, 5],    "size_min": 0.05},
            "corner_bl":       {"region": [3, 6, 7],    "size_min": 0.05},
            "corner_br":       {"region": [5, 7, 8],    "size_min": 0.05},
            "top_or_bottom":   {"size_min": 0.05},
            "left_edge":       {"region": [0, 3, 6],    "size_min": 0.04},
            "right_edge":      {"region": [2, 5, 8],    "size_min": 0.04},
        }
    elif focal_length_mm < 135:
        adjustments = {
            "center_tilted_h": {"tilt_min": 0.09, "region": None},
            "center_tilted_v": {"tilt_min": 0.08, "region": None},
            "strong_tilt":     {"tilt_min": 0.08},
            "close_up":        {"size_min": 0.12},
            # 85-135mm: chart fills frame — corners physically impossible.
            # Replace with increasing tilt angles (no region constraint).
            "corner_tl":       {"region": None, "tilt_min": 0.12, "size_min": 0.04,
                                "name": "Slight tilt", "hint": "Tilt the chart slightly in any direction"},
            "corner_tr":       {"region": None, "tilt_min": 0.18, "size_min": 0.04,
                                "name": "Moderate tilt", "hint": "Tilt the chart more — about 20°"},
            "corner_bl":       {"region": None, "tilt_min": 0.24, "size_min": 0.04,
                                "name": "Strong tilt H", "hint": "Tilt the chart strongly left or right"},
            "corner_br":       {"region": None, "tilt_min": 0.30, "size_min": 0.04,
                                "name": "Strong tilt V", "hint": "Tilt the chart strongly top or bottom"},
            "top_or_bottom":   {"region": None, "tilt_min": 0.10, "size_min": 0.04,
                                "name": "Any tilt", "hint": "Tilt the chart in any direction"},
            # left/right edge impossible at 85-135mm — use off-centre tilt variants
            "left_edge":       {"region": None, "tilt_min": 0.10, "size_min": 0.04,
                                "name": "Off-centre tilt",
                                "hint": "Tilt the chart and shift it off-centre"},
            "right_edge":      {"region": None, "tilt_min": 0.15, "size_min": 0.04,
                                "name": "Off-centre tilt 2",
                                "hint": "Tilt the chart to a different off-centre position"},
        }
    else:
        adjustments = {
            "center_tilted_h": {"tilt_min": 0.07, "region": None},
            "center_tilted_v": {"tilt_min": 0.06, "region": None},
            "strong_tilt":     {"tilt_min": 0.06},
            "close_up":        {"size_min": 0.10},
            # 135mm+: same approach — tilt-only, no region constraint
            "corner_tl":       {"region": None, "tilt_min": 0.10, "size_min": 0.025,
                                "name": "Slight tilt", "hint": "Tilt the chart slightly in any direction"},
            "corner_tr":       {"region": None, "tilt_min": 0.16, "size_min": 0.025,
                                "name": "Moderate tilt", "hint": "Tilt the chart more — about 20°"},
            "corner_bl":       {"region": None, "tilt_min": 0.22, "size_min": 0.025,
                                "name": "Strong tilt H", "hint": "Tilt the chart strongly left or right"},
            "corner_br":       {"region": None, "tilt_min": 0.28, "size_min": 0.025,
                                "name": "Strong tilt V", "hint": "Tilt the chart strongly top or bottom"},
            "top_or_bottom":   {"region": None, "tilt_min": 0.08, "size_min": 0.025,
                                "name": "Any tilt", "hint": "Tilt the chart in any direction"},
            "left_edge":       {"region": None, "tilt_min": 0.08, "size_min": 0.025,
                                "name": "Off-centre tilt",
                                "hint": "Tilt the chart and shift it off-centre"},
            "right_edge":      {"region": None, "tilt_min": 0.13, "size_min": 0.025,
                                "name": "Off-centre tilt 2",
                                "hint": "Tilt the chart to a different off-centre position"},
        }

    for pose in poses:
        update = adjustments.get(pose["id"])
        if update:
            pose.update(update)
    return poses


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


def compute_pose_metrics_sparse(
    corners: list,
    image_size: tuple,  # (width, height)
) -> dict | None:
    """Estimate pose metrics from an arbitrary sparse set of points.

    This is used for partial-board detections (e.g. ChArUco) where a full
    rows×cols grid is not available.
    """
    if not corners or len(corners) < 4:
        return None

    pts = np.array(corners, dtype=np.float64).reshape(-1, 2)
    iw, ih = image_size
    if iw <= 0 or ih <= 0:
        return None

    cx = float(pts[:, 0].mean()) / iw
    cy = float(pts[:, 1].mean()) / ih

    gx = min(2, int(cx * 3))
    gy = min(2, int(cy * 3))
    region = gy * 3 + gx

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    apparent_size = float((x_max - x_min) * (y_max - y_min)) / (iw * ih)

    # Tilt score via foreshortening: compare projected width of top vs bottom
    # half of points, and height of left vs right half.
    # Same principle as the full checkerboard metric — works on any point cloud.
    top_pts  = pts[pts[:, 1] <= np.median(pts[:, 1])]
    bot_pts  = pts[pts[:, 1] >  np.median(pts[:, 1])]
    left_pts = pts[pts[:, 0] <= np.median(pts[:, 0])]
    rgt_pts  = pts[pts[:, 0] >  np.median(pts[:, 0])]

    top_w  = float(top_pts[:, 0].max()  - top_pts[:, 0].min())  if len(top_pts)  >= 2 else 1.0
    bot_w  = float(bot_pts[:, 0].max()  - bot_pts[:, 0].min())  if len(bot_pts)  >= 2 else 1.0
    left_h = float(left_pts[:, 1].max() - left_pts[:, 1].min()) if len(left_pts) >= 2 else 1.0
    rgt_h  = float(rgt_pts[:, 1].max()  - rgt_pts[:, 1].min())  if len(rgt_pts)  >= 2 else 1.0

    w_ratio = min(top_w, bot_w)  / max(top_w, bot_w,  1e-6)
    h_ratio = min(left_h, rgt_h) / max(left_h, rgt_h, 1e-6)
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


def match_unsatisfied_pose(
    metrics: dict,
    captured_frames: list,
    required_poses: list[dict] | None = None,
) -> str | None:
    """
    Return the id of the first unsatisfied pose that the current frame satisfies,
    or None if the frame doesn't match any outstanding pose.
    Used by the auto-capture hold logic.
    """
    pose_defs = required_poses or REQUIRED_POSES
    satisfied_ids: set[str] = set()
    for frame in captured_frames:
        m = frame.get("pose_metrics")
        if not m:
            continue
        explicit_pose_id = frame.get("captured_pose_id")
        if explicit_pose_id and explicit_pose_id not in satisfied_ids:
            if any(p["id"] == explicit_pose_id for p in pose_defs):
                satisfied_ids.add(explicit_pose_id)
                continue
        for pose_def in pose_defs:
            if pose_def["id"] not in satisfied_ids and _pose_matches(m, pose_def):
                satisfied_ids.add(pose_def["id"])
                break

    for pose_def in pose_defs:
        if pose_def["id"] not in satisfied_ids and _pose_matches(metrics, pose_def):
            return pose_def["id"]
    return None


def evaluate_checklist(captured_frames: list, required_poses: list[dict] | None = None) -> dict:
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
    pose_defs = required_poses or REQUIRED_POSES
    satisfied_ids: set[str] = set()

    for frame in captured_frames:
        metrics = frame.get("pose_metrics")
        if not metrics:
            continue
        explicit_pose_id = frame.get("captured_pose_id")
        if explicit_pose_id and explicit_pose_id not in satisfied_ids:
            if any(p["id"] == explicit_pose_id for p in pose_defs):
                satisfied_ids.add(explicit_pose_id)
                continue
        for pose_def in pose_defs:
            if pose_def["id"] not in satisfied_ids and _pose_matches(metrics, pose_def):
                satisfied_ids.add(pose_def["id"])
                break

    checklist = [
        {
            "id":        p["id"],
            "name":      p["name"],
            "hint":      p["hint"],
            "satisfied": p["id"] in satisfied_ids,
        }
        for p in pose_defs
    ]

    remaining = [p for p in pose_defs if p["id"] not in satisfied_ids]
    complete = len(remaining) == 0

    next_hint = "All poses captured! Ready to calibrate." if complete else remaining[0]["hint"]
    next_pose_id = None if complete else remaining[0]["id"]

    return {
        "checklist":        checklist,
        "satisfied_count":  len(satisfied_ids),
        "total":            len(pose_defs),
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
