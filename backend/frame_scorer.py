import math
from typing import Optional

import cv2
import numpy as np

from pose_advisor import compute_pose_metrics, compute_pose_metrics_sparse

# Create once — constructing CLAHE per-frame wastes ~0.5 ms on every call
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Sony AcuTarget: 5×6 marker grid, IDs 12..41, DICT_4X4_50.
_SONY_DICT_NAME = "DICT_4X4_50"
_SONY_ID_START = 12
_SONY_GRID_W = 5
_SONY_GRID_H = 6
_SONY_MARKER_SIZE = 1.0
_SONY_MARKER_PITCH = 2.0


def score_frame(image_path: str, checkerboard_size: tuple) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return _fail(f"Cannot read image: {image_path}", 0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return _score_image(img, gray, w, h, checkerboard_size)


def score_frame_array(image: np.ndarray, checkerboard_size: tuple, fast: bool = False) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return _score_image(image, gray, w, h, checkerboard_size, fast=fast)


def _score_image(img: np.ndarray, gray: np.ndarray, w: int, h: int, checkerboard_size: tuple, fast: bool = False) -> dict:
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

    # fast=True skips the slower classic-detector fallbacks (attempts 3 & 4).
    # Safe when the board was detected in the previous frame — SB is already the
    # most accurate detector; classic is only needed for initially finding the board.
    if not fast:
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

    detection_type = "checkerboard"

    # --- Attempt 5: Sony ArUco grid fallback (partial board allowed) ---
    charuco_obj_points = []
    if not found:
        found, corners, charuco_obj_points = _detect_sony_aruco_grid(gray_eq)
        if found:
            detection_type = "aruco_grid"

    # --- Attempt 6: ChArUco fallback (partial board allowed) ---
    if not found:
        found, corners, charuco_obj_points = _detect_charuco(gray_eq, checkerboard_size)
        if found:
            detection_type = "charuco"

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
            "detection_type": None,
            "obj_points": [],
        }

    pts = corners.reshape(-1, 2)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_area = (x_max - x_min) * (y_max - y_min)
    coverage = float(bbox_area / (w * h))

    cols = checkerboard_size[0]
    first_row = pts[:min(cols, len(pts))]
    angle = _row_angle_deg(first_row)

    quality, reason = _rate(sharpness, coverage, angle)

    if detection_type == "checkerboard":
        pose_metrics = compute_pose_metrics(pts.tolist(), checkerboard_size, (w, h))
    else:
        pose_metrics = compute_pose_metrics_sparse(pts.tolist(), (w, h))

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
        "detection_type": detection_type,
        "obj_points": charuco_obj_points,
    }


def _make_charuco_board(checkerboard_size: tuple):
    """Create a ChArUco board matching the Sony calibration target.

    Sony board has 8×6 squares with ArUco markers in each square.
    If checkerboard_size is (9, 6) meaning 9×6 inner corners,
    that corresponds to an 8×6 board (where board = corners + 1 on each axis).
    """
    if not hasattr(cv2, "aruco"):
        return None
    try:
        cols, rows = int(checkerboard_size[0]), int(checkerboard_size[1])
        squares_x = cols  # 8 for Sony (was cols+1 before, but Sony is directly 8×6)
        squares_y = rows  # 6 for Sony
        square_len = 1.0
        marker_len = 0.8
        # Use DICT_5X5_100 which matches Sony board (5-bit ArUco codes)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        # Try new API first, then fallback to old API
        if hasattr(cv2.aruco, "CharucoBoard"):
            return cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
        elif hasattr(cv2.aruco, "CharucoBoard_create"):
            return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, dictionary)
    except Exception:
        pass
    return None


def _get_aruco_dictionary(dict_name: str):
    if not hasattr(cv2, "aruco"):
        return None
    if not hasattr(cv2.aruco, dict_name):
        return None
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def _detect_markers(gray: np.ndarray, dictionary):
    if dictionary is None:
        return [], None
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    return marker_corners, marker_ids


def _detect_sony_aruco_grid(gray: np.ndarray) -> tuple[bool, np.ndarray, list[list[float]]]:
    """Detect Sony 5×6 AcuTarget ArUco grid and return per-corner image/object points.

    Tries detection on multiple preprocessed versions of the image to handle
    dark, warm-tinted, or low-contrast captures from video camera outputs.
    Works when only part of the board is visible (telephoto framing).
    """
    dictionary = _get_aruco_dictionary(_SONY_DICT_NAME)
    if dictionary is None:
        return False, np.empty((0, 2), dtype=np.float32), []

    # Try several preprocessed versions — warm/dark SDI signals need extra help
    candidates = [gray]
    # Equalised
    candidates.append(_CLAHE.apply(gray))
    # Normalised stretch
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    candidates.append(norm)
    # Sharpened
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    candidates.append(sharp)

    best_pts: list[list[float]] = []
    best_obj: list[list[float]] = []

    for img in candidates:
        marker_corners, marker_ids = _detect_markers(img, dictionary)
        if marker_ids is None or len(marker_ids) < 2:
            continue

        pts: list[list[float]] = []
        obj: list[list[float]] = []
        for idx, marker_id in enumerate(marker_ids.flatten().tolist()):
            mid = int(marker_id)
            rel = mid - _SONY_ID_START
            if rel < 0 or rel >= (_SONY_GRID_W * _SONY_GRID_H):
                continue
            col = rel % _SONY_GRID_W
            row = rel // _SONY_GRID_W
            img4 = np.array(marker_corners[idx], dtype=np.float32).reshape(4, 2)
            # Use marker centre — avoids rotation-order ambiguity across markers
            center = img4.mean(axis=0)
            pts.append(center.tolist())
            obj.append([float(col) * _SONY_MARKER_PITCH, float(row) * _SONY_MARKER_PITCH, 0.0])

        if len(pts) > len(best_pts):
            best_pts, best_obj = pts, obj

    if len(best_pts) < 4:
        return False, np.empty((0, 2), dtype=np.float32), []
    return True, np.array(best_pts, dtype=np.float32), best_obj


def _charuco_obj_points_from_ids(board, ids: np.ndarray) -> list[list[float]]:
    if ids is None or len(ids) == 0:
        return []
    ids_flat = ids.flatten().astype(int)
    if hasattr(board, "getChessboardCorners"):
        all_corners = board.getChessboardCorners()
    else:
        all_corners = board.chessboardCorners
    all_corners = np.array(all_corners, dtype=np.float32).reshape(-1, 3)
    out = []
    for cid in ids_flat:
        if 0 <= cid < len(all_corners):
            out.append(all_corners[cid].tolist())
    return out


def _detect_charuco(gray: np.ndarray, checkerboard_size: tuple) -> tuple[bool, np.ndarray, list[list[float]]]:
    board = _make_charuco_board(checkerboard_size)
    if board is None:
        return False, np.empty((0, 2), dtype=np.float32), []
    try:
        # Get dictionary from board if available, otherwise use a default
        if hasattr(board, "getDictionary"):
            dictionary = board.getDictionary()
        else:
            # Fallback: create fresh dictionary
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker_corners, marker_ids = _detect_markers(gray, dictionary)
        if marker_ids is None or len(marker_ids) < 2:
            return False, np.empty((0, 2), dtype=np.float32), []
        # Interpolate ChArUco corners from detected markers
        charuco_corners, charuco_ids, rejected_charuco = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            return False, np.empty((0, 2), dtype=np.float32), []
        pts = np.array(charuco_corners, dtype=np.float32).reshape(-1, 2)
        obj = _charuco_obj_points_from_ids(board, np.array(charuco_ids))
        if len(obj) != len(pts) or len(obj) < 4:
            return False, np.empty((0, 2), dtype=np.float32), []
        return True, pts, obj
    except Exception as e:
        return False, np.empty((0, 2), dtype=np.float32), []


def _row_angle_deg(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return abs(math.degrees(math.atan2(dy, dx)))


def _rate(sharpness: float, coverage: float, angle: Optional[float]) -> tuple[str, str]:
    # Hard fail: genuinely unusable (very extreme blur)
    if sharpness < 15:
        return "fail", f"extremely blurry (Laplacian {sharpness:.0f})"
    # Soft warns — pose system handles position/tilt requirements
    reasons = []
    if sharpness < 50:
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
    }
