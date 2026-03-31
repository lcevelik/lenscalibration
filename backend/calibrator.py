import numpy as np
import cv2


def run_calibration(
    scored_frames: list,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    image_size: tuple,  # (width, height)
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
            f"({skipped} skipped as 'fail')"
        )

    # --- Object points (3-D grid, z = 0) ------------------------------------
    objp = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    objp[:, :2] = (
        np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_mm
    )

    obj_points = []
    img_points = []
    paths = []

    for frame in usable:
        corners = np.array(frame["corners"], dtype=np.float32).reshape(-1, 1, 2)
        if corners.shape[0] != board_rows * board_cols:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        paths.append(frame.get("path", ""))

    if len(obj_points) < 3:
        return _error("Not enough frames with the correct corner count after filtering")

    # --- Calibration --------------------------------------------------------
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # --- FOV ----------------------------------------------------------------
    aperture_w, aperture_h = 1.0, 1.0  # sensor size unknown; use 1mm as neutral
    fov_x, fov_y, _, _, aspect = cv2.calibrationMatrixValues(
        camera_matrix, image_size, aperture_w, aperture_h
    )

    # --- Per-image reprojection error ---------------------------------------
    per_image_errors = []
    for i, (obj, img, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        err = float(
            np.sqrt(np.mean(np.sum((img.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1)))
        )
        per_image_errors.append({"path": paths[i], "error": round(err, 4), "outlier": False})

    mean_err = sum(e["error"] for e in per_image_errors) / len(per_image_errors)
    outlier_threshold = mean_err * 1.5
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
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence(rms: float) -> str:
    if rms < 0.3:
        return "excellent"
    if rms < 0.5:
        return "good"
    if rms < 1.0:
        return "marginal"
    return "poor"


def _error(reason: str) -> dict:
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
        "error": reason,
    }
