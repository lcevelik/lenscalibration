import math
from typing import Optional

import cv2
import numpy as np

from pose_advisor import compute_pose_metrics


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

    # --- Primary detector: findChessboardCornersSB (saddle-point, sub-pixel built in) ---
    found, corners = cv2.findChessboardCornersSB(gray, checkerboard_size, cv2.CALIB_CB_NORMALIZE_IMAGE)

    # --- Fallback: classic detector with quad filtering ---
    if not found:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FILTER_QUADS
        )
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not found:
        quality, reason = _rate(sharpness, coverage=0.0, angle=None)
        return {
            "found": False,
            "corners": [],
            "sharpness": round(sharpness, 2),
            "coverage": 0.0,
            "angle": None,
            "quality": quality,
            "reason": reason if reason else "Checkerboard not detected",
            "image_width": w,
            "image_height": h,
            "pose_metrics": None,
        }

    pts = corners.reshape(-1, 2)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_area = (x_max - x_min) * (y_max - y_min)
    coverage = float(bbox_area / (w * h))

    cols = checkerboard_size[0]
    first_row = pts[:cols]
    angle = _row_angle_deg(first_row)

    quality, reason = _rate(sharpness, coverage, angle)

    pose_metrics = compute_pose_metrics(pts.tolist(), checkerboard_size, (w, h))

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
    }


def _row_angle_deg(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return abs(math.degrees(math.atan2(dy, dx)))


def _rate(sharpness: float, coverage: float, angle: Optional[float]) -> tuple[str, str]:
    reasons = []
    if sharpness < 100:
        reasons.append(f"blurry (Laplacian {sharpness:.0f} < 100)")
    if coverage < 0.40:
        reasons.append(f"low coverage ({coverage * 100:.1f}% < 40%)")
    if angle is not None and angle < 10:
        reasons.append(f"board too flat ({angle:.1f}° < 10°)")
    if not reasons:
        return "good", "All metrics pass"
    quality = "fail" if len(reasons) >= 2 else "warn"
    return quality, "; ".join(reasons)


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
    }
