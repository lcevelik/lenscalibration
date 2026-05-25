"""
Shared calibration helpers used by both calibrator.py and zoom_calibrator.py.

Extracted to eliminate duplication and ensure consistent behaviour across
single-FL and multi-FL calibration paths.
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Object-point grid
# ---------------------------------------------------------------------------

def make_objp(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """Build a planar (z=0) object-point array for a cols×rows inner-corner grid."""
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return pts


# ---------------------------------------------------------------------------
# Confidence grading
# ---------------------------------------------------------------------------

def confidence(rms: float, sparse: bool = False) -> str:
    """Grade calibration quality. Sparse-board (ArUco) tolerates higher RMS."""
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


# ---------------------------------------------------------------------------
# Plausibility check
# ---------------------------------------------------------------------------

def is_implausible_solution(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    calib_size: tuple,
) -> bool:
    """Return True if calibration results are clearly non-physical."""
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


# ---------------------------------------------------------------------------
# Constrained fallback solver
# ---------------------------------------------------------------------------

def run_constrained_fallback(
    obj_points: list,
    img_points: list,
    calib_size: tuple,
    fx_init: float | None = None,
) -> tuple:
    """Fallback solve with stronger regularization for tele / low-parallax sets.

    Parameters
    ----------
    fx_init : optional initial focal length in pixels.  When provided (e.g. from
              sensor_width_mm and fl_mm), the solver starts closer to the true
              solution, improving convergence for telephoto groups.

    Returns the standard (rms, camera_matrix, dist_coeffs, rvecs, tvecs) tuple.
    """
    w, h = calib_size
    fx0 = fx_init if fx_init and fx_init > 0 else float(max(w, h))
    cam0 = np.array(
        [[fx0, 0.0, w / 2.0],
         [0.0, fx0, h / 2.0],
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
