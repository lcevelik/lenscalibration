import numpy as np
import cv2
from collections import Counter


def _run_calibration_pass(
    obj_points: list,
    img_points: list,
    calib_size: tuple,
    sparse_mode: bool = False,
):
    calib_flags = getattr(cv2, "CALIB_FIX_SKEW", 0)
    if not sparse_mode:
        return cv2.calibrateCamera(obj_points, img_points, calib_size, None, None, flags=calib_flags)

    # Sparse correspondences (e.g. marker centers) are under-constrained for
    # high-order distortion terms. Use a conservative model to prevent overfit.
    w, h = calib_size
    cam0 = np.array(
        [[max(w, h), 0.0, w / 2.0],
         [0.0, max(w, h), h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dc0 = np.zeros((5, 1), dtype=np.float64)
    calib_flags |= (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_ZERO_TANGENT_DIST
    )
    return cv2.calibrateCamera(obj_points, img_points, calib_size, cam0, dc0, flags=calib_flags)


def _run_constrained_fallback_pass(obj_points: list, img_points: list, calib_size: tuple):
    """Fallback solve with stronger regularization for tele / low-parallax sets."""
    w, h = calib_size
    cam0 = np.array(
        [[max(w, h), 0.0, w / 2.0],
         [0.0, max(w, h), h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dc0 = np.zeros((5, 1), dtype=np.float64)
    flags = getattr(cv2, "CALIB_FIX_SKEW", 0)
    flags |= (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_ASPECT_RATIO
    )
    return cv2.calibrateCamera(obj_points, img_points, calib_size, cam0, dc0, flags=flags)


def _is_implausible_solution(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, calib_size: tuple) -> bool:
    w, h = calib_size
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    dc = dist_coeffs.flatten()
    k1 = float(dc[0]) if len(dc) > 0 else 0.0
    k2 = float(dc[1]) if len(dc) > 1 else 0.0
    p1 = float(dc[2]) if len(dc) > 2 else 0.0
    p2 = float(dc[3]) if len(dc) > 3 else 0.0

    if not np.isfinite([fx, fy, cx, cy, k1, k2, p1, p2]).all():
        return True
    if fx < 0.2 * w or fy < 0.2 * h:
        return True
    if fx > 20.0 * w or fy > 20.0 * h:
        return True
    # Principal point must lie within ±75 % of the image half-dimension from
    # the image centre.  Testing against absolute pixel values (old check:
    # abs(cx) > 3*w) measures from the top-left corner and misses cases like
    # cy = -314 for a 1080 px tall frame.
    if abs(cx - w / 2) > 0.75 * w or abs(cy - h / 2) > 0.75 * h:
        return True
    if abs(k1) > 2.0 or abs(k2) > 2.0:
        return True
    if abs(p1) > 0.2 or abs(p2) > 0.2:
        return True
    return False


def _compute_per_image_errors(
    obj_points: list,
    img_points: list,
    paths: list,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: list,
    tvecs: list,
) -> list[dict]:
    per_image_errors = []
    for i, (obj, img, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        diff = img.reshape(-1, 2) - projected.reshape(-1, 2)
        sq_err = np.sum(diff ** 2, axis=1)
        if np.any(~np.isfinite(sq_err)):
            err = float("inf")
        else:
            err = float(np.sqrt(np.mean(sq_err)))
        per_image_errors.append({"path": paths[i], "error": round(err, 4), "outlier": False})
    return per_image_errors


def _mark_outliers(per_image_errors: list[dict]) -> tuple[list[dict], float]:
    errors_arr = np.array([e["error"] for e in per_image_errors])
    q1, q3 = float(np.percentile(errors_arr, 25)), float(np.percentile(errors_arr, 75))
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr if iqr > 0 else float(np.median(errors_arr)) * 2.0
    for entry in per_image_errors:
        entry["outlier"] = entry["error"] > outlier_threshold
    return per_image_errors, outlier_threshold


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

    # Use one detection model per solve. Mixing checkerboard and aruco-grid
    # frames in one calibration causes inconsistent board coordinates.
    # Select the detection type with the most usable frames (minimum 3).
    # This ensures that on ChArUco boards (e.g. Sony AcuTarget), where ArUco
    # detection succeeds on more frames than the checkerboard detector, we use
    # the larger frame set even if checkerboard technically passed the minimum.
    det_counter = Counter((f.get("detection_type") or "checkerboard") for f in usable)
    qualified = [d for d in det_counter if det_counter[d] >= 3]
    if qualified:
        selected_det = max(qualified, key=lambda d: det_counter[d])
    else:
        selected_det = det_counter.most_common(1)[0][0]
    usable = [f for f in usable if (f.get("detection_type") or "checkerboard") == selected_det]

    # --- Object points (3-D grid, z = 0) ------------------------------------
    # Full-board template; partial frames get their own per-frame objp via
    # the partial_grid_size field written by frame_scorer.
    full_objp = _make_objp(board_cols, board_rows, square_size_mm)

    obj_points = []
    img_points = []
    paths = []

    sparse_mode = False
    for frame in usable:
        corners = np.array(frame["corners"], dtype=np.float32).reshape(-1, 1, 2)
        frame_obj_points = frame.get("obj_points")
        partial_size = frame.get("partial_grid_size")
        if frame_obj_points:
            obj = np.array(frame_obj_points, dtype=np.float32).reshape(-1, 3)
            if obj.shape[0] != corners.shape[0] or obj.shape[0] < 6:
                continue
            if (frame.get("detection_type") or "") == "charuco":
                # ChArUco obj points come back in board-square units (1 = 1 square)
                # from OpenCV's getChessboardCorners(); scale to mm.
                obj = obj * float(square_size_mm)
            # aruco_grid obj points are already in mm (physical mm from the
            # Sony board lookup table), so no further scaling is needed.
            sparse_mode = True
        elif partial_size:
            p_cols, p_rows = int(partial_size[0]), int(partial_size[1])
            if corners.shape[0] != p_cols * p_rows:
                continue
            obj = _make_objp(p_cols, p_rows, square_size_mm)
        else:
            if corners.shape[0] != board_rows * board_cols:
                continue
            obj = full_objp
        if squeeze_ratio > 1.0:
            corners = corners.copy()
            corners[:, :, 0] *= squeeze_ratio  # scale x to de-squeezed space
        obj_points.append(obj)
        img_points.append(corners)
        paths.append(frame.get("path", ""))

    if len(obj_points) < 3:
        return _error("Not enough frames with the correct corner count after filtering", squeeze_ratio)

    # --- Calibration --------------------------------------------------------
    calib_size = (int(image_size[0] * squeeze_ratio), image_size[1]) if squeeze_ratio > 1.0 else image_size
    solver_warning = None
    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = _run_calibration_pass(
            obj_points, img_points, calib_size, sparse_mode=sparse_mode
        )
    except cv2.error as e:
        return _error(f"OpenCV calibration failed: {e}", squeeze_ratio)

    if _is_implausible_solution(camera_matrix, dist_coeffs, calib_size):
        try:
            rms_fb, cm_fb, dc_fb, rv_fb, tv_fb = _run_constrained_fallback_pass(
                obj_points, img_points, calib_size
            )
            if not _is_implausible_solution(cm_fb, dc_fb, calib_size):
                rms, camera_matrix, dist_coeffs, rvecs, tvecs = rms_fb, cm_fb, dc_fb, rv_fb, tv_fb
                solver_warning = (
                    "Used constrained tele fallback solver (k2/k3 fixed, tangential fixed). "
                    "Add wider-FOV / stronger-tilt frames for highest accuracy."
                )
            else:
                return _error(
                    "Calibration solution is unstable/implausible for this frame set. "
                    "Capture more varied poses (different positions/tilts) or include wider-FOV frames.",
                    squeeze_ratio,
                )
        except cv2.error:
            return _error(
                "Calibration solution is unstable/implausible for this frame set. "
                "Capture more varied poses (different positions/tilts) or include wider-FOV frames.",
                squeeze_ratio,
            )

    if not np.isfinite(rms):
        return _error(
            f"Calibration RMS too high ({rms:.2f}px). Data is inconsistent for this board/profile.",
            squeeze_ratio,
        )

    if rms > 5.0:
        if sparse_mode:
            if rms > 20.0:
                return _error(
                    f"Calibration RMS too high ({rms:.2f}px) even for sparse ArUco mode. "
                    "ArUco marker points are too inconsistent — capture more frames with the board "
                    "at varied angles/positions covering more of the frame, and ensure the target "
                    "is sharp and well-lit.",
                    squeeze_ratio,
                )
            relax_msg = (
                f"High RMS ({rms:.2f}px) in sparse tele mode. "
                "Returned a low-confidence fit for continuity; distortion is likely unreliable."
            )
            solver_warning = f"{solver_warning} {relax_msg}" if solver_warning else relax_msg
        else:
            return _error(
                f"Calibration RMS too high ({rms:.2f}px). Data is inconsistent for this board/profile.",
                squeeze_ratio,
            )

    # --- FOV ----------------------------------------------------------------
    aperture_w, aperture_h = 1.0, 1.0  # sensor size unknown; use 1mm as neutral
    fov_x, fov_y, _, _, aspect = cv2.calibrationMatrixValues(
        camera_matrix, calib_size, aperture_w, aperture_h
    )

    # --- Per-image reprojection error ---------------------------------------
    per_image_errors = _compute_per_image_errors(
        obj_points, img_points, paths, camera_matrix, dist_coeffs, rvecs, tvecs
    )
    per_image_errors, _ = _mark_outliers(per_image_errors)

    # If a few frames are clear outliers, rerun calibration using inliers only.
    # This improves stability without requiring the operator to manually prune frames.
    inlier_indices = [i for i, entry in enumerate(per_image_errors) if not entry["outlier"]]
    if 3 <= len(inlier_indices) < len(obj_points):
        inlier_obj_points = [obj_points[i] for i in inlier_indices]
        inlier_img_points = [img_points[i] for i in inlier_indices]
        inlier_paths = [paths[i] for i in inlier_indices]
        try:
            refined_rms, refined_camera_matrix, refined_dist_coeffs, refined_rvecs, refined_tvecs = _run_calibration_pass(
                inlier_obj_points, inlier_img_points, calib_size, sparse_mode=sparse_mode
            )
            if np.isfinite(refined_rms) and refined_rms <= rms:
                rms = refined_rms
                camera_matrix = refined_camera_matrix
                dist_coeffs = refined_dist_coeffs
                paths = inlier_paths
                obj_points = inlier_obj_points
                img_points = inlier_img_points
                rvecs = refined_rvecs
                tvecs = refined_tvecs
                per_image_errors = _compute_per_image_errors(
                    obj_points, img_points, paths, camera_matrix, dist_coeffs, rvecs, tvecs
                )
                per_image_errors, _ = _mark_outliers(per_image_errors)
        except cv2.error:
            pass

    # --- Confidence ---------------------------------------------------------
    confidence = _confidence(rms, sparse=sparse_mode)

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
        "detection_type": selected_det,
        "squeeze_ratio": squeeze_ratio,
        "lens_type": "anamorphic" if squeeze_ratio > 1.0 else "spherical",
        "warning": solver_warning,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_objp(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """Build a planar (z=0) object-point array for a cols×rows inner-corner grid."""
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return pts


def _confidence(rms: float, sparse: bool = False) -> str:
    if sparse:
        if rms < 0.7:  return "excellent"
        if rms < 1.5:  return "good"
        if rms < 3.0:  return "marginal"
        return "poor"
    else:
        if rms < 0.3:  return "excellent"
        if rms < 0.5:  return "good"
        if rms < 1.0:  return "marginal"
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
