import math
from typing import Optional

import cv2
import numpy as np

from pose_advisor import compute_pose_metrics

# Create once — constructing CLAHE per-frame wastes ~0.5 ms on every call
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _partial_candidates(full_cols: int, full_rows: int) -> list:
    """Return candidate sub-grid sizes (cols, rows) for partial board detection.

    Generates all valid reductions in steps of 2 (to keep even checker patterns),
    sorted by area descending so we try the largest visible sub-grid first.
    Minimum viable size is (4, 3).
    """
    candidates = []
    for dc in range(0, full_cols - 3, 2):
        for dr in range(0, full_rows - 2, 2):
            if dc == 0 and dr == 0:
                continue  # skip the full size — already tried
            c, r = full_cols - dc, full_rows - dr
            if c >= 4 and r >= 3:
                candidates.append((c, r))
    candidates.sort(key=lambda x: -(x[0] * x[1]))
    seen: set = set()
    unique: list = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:16]  # cap attempts to avoid excessive runtime


def score_frame(image_path: str, checkerboard_size: tuple) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return _fail(f"Cannot read image: {image_path}", 0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return _score_image(img, gray, w, h, checkerboard_size)


def score_frame_array(image: np.ndarray, checkerboard_size: tuple) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return _score_image(image, gray, w, h, checkerboard_size)


def _score_image(img: np.ndarray, gray: np.ndarray, w: int, h: int, checkerboard_size: tuple) -> dict:
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # CLAHE-enhanced gray for detection (improves contrast on SDI/capture-card signals)
    gray_eq = _CLAHE.apply(gray)

    # --- Attempt 1: SB on CLAHE-enhanced (exhaustive + accuracy) ---
    sb_flags = (cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY)
    found, corners = cv2.findChessboardCornersSB(gray_eq, checkerboard_size, sb_flags)

    # --- Attempt 2: SB on raw gray ---
    if not found:
        found, corners = cv2.findChessboardCornersSB(
            gray, checkerboard_size, cv2.CALIB_CB_NORMALIZE_IMAGE)

    # --- Attempt 3: classic detector on CLAHE-enhanced gray ---
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FILTER_QUADS)
        found, corners = cv2.findChessboardCorners(gray_eq, checkerboard_size, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Refine on the same image the detector used (gray_eq) for consistency
            corners = cv2.cornerSubPix(gray_eq, corners, (11, 11), (-1, -1), criteria)

    # --- Attempt 4: classic detector on raw gray ---
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FILTER_QUADS)
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # --- Attempt 5: partial sub-grid (telephoto / chart overflow) ---
    # When the chart fills more than the frame at long focal lengths, try
    # progressively smaller inner-corner grids.  The detected sub-grid is
    # treated as an independent planar target; the partial_grid_size field
    # lets the calibrators build the correct object-point set per frame.
    partial_size: Optional[tuple] = None
    if not found:
        for sub_size in _partial_candidates(checkerboard_size[0], checkerboard_size[1]):
            sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(gray_eq, sub_size, sb_flags)
            if found and corners is not None:
                partial_size = sub_size
                break
            found, corners = cv2.findChessboardCornersSB(
                gray, sub_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found and corners is not None:
                partial_size = sub_size
                break

    if not found:
        return {
            "found": False,
            "corners": [],
            "sharpness": round(sharpness, 2),
            "coverage": 0.0,
            "angle": None,
            "quality": "fail",
            "reason": "Checkerboard not detected",
            "image_width": w,
            "image_height": h,
            "pose_metrics": None,
            "partial_grid_size": None,
        }

    # When using a partial sub-grid, use its actual dimensions for angle/coverage
    effective_size = partial_size if partial_size else checkerboard_size
    pts = corners.reshape(-1, 2)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_area = (x_max - x_min) * (y_max - y_min)
    coverage = float(bbox_area / (w * h))

    cols = effective_size[0]
    first_row = pts[:cols]
    angle = _row_angle_deg(first_row)

    quality, reason = _rate(sharpness, coverage, angle)
    if partial_size:
        reason = f"partial board {partial_size[0]}×{partial_size[1]}; " + reason

    pose_metrics = compute_pose_metrics(pts.tolist(), effective_size, (w, h))

    return {
        "found": True,
        "corners": pts.tolist(),
        "sharpness": round(sharpness, 2),
        "coverage": round(coverage, 4),
        "angle": round(angle, 2),
        "quality": quality,
        "reason": reason,
        "image_width": w,
        "image_height": h,
        "pose_metrics": pose_metrics,
        "partial_grid_size": list(partial_size) if partial_size else None,
    }


def _row_angle_deg(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return abs(math.degrees(math.atan2(dy, dx)))


def _rate(sharpness: float, coverage: float, angle: Optional[float]) -> tuple[str, str]:
    # Hard fail: genuinely too blurry to use
    if sharpness < 30:
        return "fail", f"too blurry (Laplacian {sharpness:.0f})"
    # Soft warns — pose system handles position/tilt requirements
    reasons = []
    if sharpness < 80:
        reasons.append(f"slightly blurry ({sharpness:.0f})")
    if coverage < 0.04:
        reasons.append(f"board very small ({coverage * 100:.1f}%)")
    if not reasons:
        return "good", "All metrics pass"
    return "warn", "; ".join(reasons)


def _fail(reason: str, w: int, h: int) -> dict:
    return {
        "found": False,
        "corners": [],
        "sharpness": 0.0,
        "coverage": 0.0,
        "angle": None,
        "quality": "fail",
        "reason": reason,
        "image_width": w,
        "image_height": h,
        "pose_metrics": None,
        "partial_grid_size": None,
    }
