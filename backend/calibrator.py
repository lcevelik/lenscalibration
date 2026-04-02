import numpy as np
import cv2


def run_calibration(
    scored_frames: list,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    image_size: tuple,  # (width, height)
    squeeze_ratio: float = 1.0,
) -> dict:
    """
    Run OpenCV camera calibration from pre-scored frames.

    Parameters
    ----------
    scored_frames   : list of dicts from frame_scorer.score_frame(), each must
                      have 'corners', 'quality', and 'path' keys
    board_cols      : inner corner count along columns
    board_rows      : inner corner count along rows
    square_size_mm  : physical size of one square in millimetres
    image_size      : (width, height) in pixels — must be consistent across frames

    Returns
    -------
    {
        rms               : float,
        camera_matrix     : list[list[float]],
        dist_coeffs       : list[float],
        fov_x             : float,
        fov_y             : float,
        per_image_errors  : [{"path": str, "error": float, "outlier": bool}],
        confidence        : "excellent" | "good" | "marginal" | "poor",
        used_frames       : int,
        skipped_frames    : int,
    }
    """
    usable = [f for f in scored_frames if f.get("quality") != "fail" and f.get("corners")]
    skipped = len(scored_frames) - len(usable)

    if len(usable) < 3:
        return _error(
            f"Need at least 3 usable frames, got {len(usable)} "
            f"({skipped} skipped as 'fail')",
            squeeze_ratio,
        )

    # --- Object points (3-D grid, z = 0) ------------------------------------
    # Full-board template; partial frames get their own per-frame objp via
    # the partial_grid_size field written by frame_scorer.
    full_objp = _make_objp(board_cols, board_rows, square_size_mm)

    obj_points = []
    img_points = []
    paths = []

    for frame in usable:
        corners = np.array(frame["corners"], dtype=np.float32).reshape(-1, 1, 2)
        partial_size = frame.get("partial_grid_size")
        if partial_size:
            p_cols, p_rows = int(partial_size[0]), int(partial_size[1])
            if corners.shape[0] != p_cols * p_rows:
                continue
            frame_objp = _make_objp(p_cols, p_rows, square_size_mm)
        else:
            if corners.shape[0] != board_rows * board_cols:
                continue
            frame_objp = full_objp
        if squeeze_ratio > 1.0:
            corners = corners.copy()
            corners[:, :, 0] *= squeeze_ratio  # scale x to de-squeezed space
        obj_points.append(frame_objp)
        img_points.append(corners)
        paths.append(frame.get("path", ""))

    if len(obj_points) < 3:
        return _error("Not enough frames with the correct corner count after filtering", squeeze_ratio)

    # --- Calibration --------------------------------------------------------
    calib_size = (int(image_size[0] * squeeze_ratio), image_size[1]) if squeeze_ratio > 1.0 else image_size
    # CALIB_FIX_SKEW: modern digital sensors have zero pixel skew; estimating it
    # with limited data overfits and degrades accuracy.
    calib_flags = cv2.CALIB_FIX_SKEW
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, calib_size, None, None, flags=calib_flags
    )

    # --- FOV ----------------------------------------------------------------
    aperture_w, aperture_h = 1.0, 1.0  # sensor size unknown; use 1mm as neutral
    fov_x, fov_y, _, _, aspect = cv2.calibrationMatrixValues(
        camera_matrix, calib_size, aperture_w, aperture_h
    )

    # --- Per-image reprojection error ---------------------------------------
    per_image_errors = []
    for i, (obj, img, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        diff = img.reshape(-1, 2) - projected.reshape(-1, 2)
        sq_err = np.sum(diff ** 2, axis=1)
        # Guard against NaN/Inf from degenerate pose estimates
        if np.any(~np.isfinite(sq_err)):
            err = float("inf")
        else:
            err = float(np.sqrt(np.mean(sq_err)))
        per_image_errors.append({"path": paths[i], "error": round(err, 4), "outlier": False})

    # Use Tukey IQR method — robust against skewed distributions caused by one
    # very bad frame dragging the mean up and masking all other outliers.
    errors_arr = np.array([e["error"] for e in per_image_errors])
    q1, q3 = float(np.percentile(errors_arr, 25)), float(np.percentile(errors_arr, 75))
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr if iqr > 0 else float(np.median(errors_arr)) * 2.0
    for entry in per_image_errors:
        entry["outlier"] = entry["error"] > outlier_threshold

    # --- Confidence ---------------------------------------------------------
    confidence = _confidence(rms)

    return {
        "rms": round(rms, 4),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "fov_x": round(fov_x, 4),
        "fov_y": round(fov_y, 4),
        "per_image_errors": per_image_errors,
        "confidence": confidence,
        "used_frames": len(obj_points),
        "skipped_frames": skipped,
        "squeeze_ratio": squeeze_ratio,
        "lens_type": "anamorphic" if squeeze_ratio > 1.0 else "spherical",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_objp(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """Build a planar (z=0) object-point array for a cols×rows inner-corner grid."""
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return pts


def _confidence(rms: float) -> str:
    if rms < 0.3:
        return "excellent"
    if rms < 0.5:
        return "good"
    if rms < 1.0:
        return "marginal"
    return "poor"


def _error(reason: str, squeeze_ratio: float = 1.0) -> dict:
    return {
        "rms": None,
        "camera_matrix": None,
        "dist_coeffs": None,
        "fov_x": None,
        "fov_y": None,
        "per_image_errors": [],
        "confidence": "poor",
        "used_frames": 0,
        "skipped_frames": 0,
        "squeeze_ratio": squeeze_ratio,
        "lens_type": "anamorphic" if squeeze_ratio > 1.0 else "spherical",
        "error": reason,
    }
