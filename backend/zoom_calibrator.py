"""
Zoom sweep calibration: calibrate at multiple focal lengths and estimate
per-FL nodal offsets from optical-center geometry.

Working-distance guidance
-------------------------
For accurate nodal offsets across the full zoom range the user should keep
the calibration chart at a consistent *angular size* in the frame — not at a
fixed physical distance.  At longer focal lengths, step the camera (or chart)
back so the chart fills roughly the same fraction of the frame:

    d(f) = d_wide × (f / f_wide)

When each FL group carries a ``working_distance_mm`` field, the backend
subtracts the per-FL physical distance from the raw optical-centre Z so that
all groups share the same body-fixed coordinate origin.  Without this field
(or when it is 0) the legacy behaviour is preserved (assumes same distance).

Partial-board frames
--------------------
Frames where only a sub-region of the chart is visible (telephoto overflow)
carry a ``partial_grid_size`` field set by frame_scorer.  The calibrator uses
that sub-grid size to build the correct per-frame object-point array instead
of the full-board template.

Nodal extrapolation
-------------------
After PCHIP interpolation of measured points, a Padé (2,1) rational model is
fitted to the measured nodal offsets and used to extend the table beyond the
calibrated FL range (up to the outermost *requested* FL).
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator  # monotonic cubic spline

from nodal_model import fit_nodal_model, predict_nodal


def run_zoom_calibration(
    fl_groups: list,        # [{focal_length_mm, frames: [scored_frame, ...]}, ...]
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    image_size: tuple,      # (width, height)
    sensor_width_mm: float = 0.0,
    sensor_height_mm: float = 0.0,
    squeeze_ratio: float = 1.0,
) -> dict:
    """
    Run a separate calibration for each focal-length group and derive
    per-FL nodal offsets from the mean optical-centre position.

    Each group dict may include an optional ``working_distance_mm`` key.
    When present, the raw optical-centre Z is corrected by subtracting the
    physical camera-to-chart distance so that nodal offsets across groups
    measured at different distances are directly comparable.

    Returns
    -------
    {
        success: bool,
        error: str | None,
        fl_results: [
          {focal_length_mm, rms, fx_px, fy_px, cx_px, cy_px,
           focal_length_computed_mm, dist_coeffs, camera_matrix,
           optical_center_world:[x,y,z], used_frames,
           per_image_errors, confidence, error?}, ...
        ],
        nodal_offsets_mm: {str(fl_mm): float, ...},   # Z-shift vs best-RMS FL
        nodal_model: dict | None,                      # Padé/poly model for export
    }
    """
    full_objp = _make_objp(board_cols, board_rows, square_size_mm)

    fl_results: list[dict] = []
    optical_centers: list          = []   # None or np.ndarray (corrected) per group
    K_best: np.ndarray | None      = None  # camera matrix from lowest-RMS FL so far
    D_best: np.ndarray | None      = None  # dist coeffs from lowest-RMS FL so far

    for group in fl_groups:
        fl_mm           = float(group["focal_length_mm"])
        frames          = group.get("frames", [])
        working_dist_mm = float(group.get("working_distance_mm", 0.0))
        usable          = [f for f in frames
                           if f.get("quality") != "fail" and f.get("corners")]

        if len(usable) < 3:
            fl_results.append({
                "focal_length_mm": fl_mm,
                "rms": None,
                "error": f"Need ≥ 3 usable frames, got {len(usable)}",
            })
            optical_centers.append(None)
            continue

        obj_points, img_points, paths = [], [], []
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
                corners[:, :, 0] *= squeeze_ratio
            obj_points.append(frame_objp)
            img_points.append(corners)
            paths.append(frame.get("path", ""))

        if len(obj_points) < 3:
            # Not enough frames even after corner-count validation.
            # If we have a K from a previous FL, attempt pose-only estimation.
            pose_result = _pose_only_calibration(
                usable, K_best, D_best, board_cols, board_rows,
                square_size_mm, squeeze_ratio, working_dist_mm, fl_mm,
            )
            fl_results.append(pose_result)
            optical_centers.append(pose_result.get("_mean_center"))
            continue

        calib_size = (int(image_size[0] * squeeze_ratio), image_size[1]) if squeeze_ratio > 1.0 else image_size
        calib_flags = cv2.CALIB_FIX_SKEW
        try:
            rms, cam_mtx, dist_c, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, calib_size, None, None, flags=calib_flags
            )
        except Exception as exc:
            # calibrateCamera failed (e.g. all-partial, poorly conditioned).
            # Fall back to pose-only path with fixed K from nearest calibrated FL.
            pose_result = _pose_only_calibration(
                usable, K_best, D_best, board_cols, board_rows,
                square_size_mm, squeeze_ratio, working_dist_mm, fl_mm,
            )
            if pose_result.get("rms") is None:
                pose_result["error"] = str(exc)
            fl_results.append(pose_result)
            optical_centers.append(pose_result.get("_mean_center"))
            continue

        fx_px = float(cam_mtx[0, 0])
        fy_px = float(cam_mtx[1, 1])
        cx_px = float(cam_mtx[0, 2])
        cy_px = float(cam_mtx[1, 2])
        dc    = dist_c.flatten()

        fl_computed = (fx_px * sensor_width_mm / calib_size[0]
                       if sensor_width_mm > 0 else 0.0)

        # ── Optical centre for each image ──────────────────────────────────
        centers = []
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            centers.append((-R.T @ tv).flatten())
        mean_center = np.mean(centers, axis=0).copy()

        # Working-distance correction: when the user stepped back to keep
        # the chart visible at this FL, subtract the extra distance so all
        # groups are expressed in the same body-fixed coordinate frame.
        if working_dist_mm > 0:
            mean_center[2] -= working_dist_mm

        optical_centers.append(mean_center)

        # Track best K for fallback use in subsequent (harder) FL groups
        if K_best is None or rms < (fl_results[-1].get("rms") or float("inf") if fl_results else float("inf")):
            K_best = cam_mtx.copy()
            D_best = dist_c.copy()

        # ── Per-image reprojection errors ──────────────────────────────────
        pie = []
        for i, (obj, img, rv, tv) in enumerate(
                zip(obj_points, img_points, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(obj, rv, tv, cam_mtx, dist_c)
            err = float(np.sqrt(np.mean(
                np.sum((img.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1)
            )))
            pie.append({"path": paths[i], "error": round(err, 4), "outlier": False})

        pie_arr = np.array([e["error"] for e in pie])
        q1, q3 = float(np.percentile(pie_arr, 25)), float(np.percentile(pie_arr, 75))
        iqr = q3 - q1
        pie_threshold = q3 + 1.5 * iqr if iqr > 0 else float(np.median(pie_arr)) * 2.0
        for entry in pie:
            entry["outlier"] = entry["error"] > pie_threshold

        fl_results.append({
            "focal_length_mm":          fl_mm,
            "rms":                      round(rms, 4),
            "fx_px":                    round(fx_px, 2),
            "fy_px":                    round(fy_px, 2),
            "cx_px":                    round(cx_px, 2),
            "cy_px":                    round(cy_px, 2),
            "focal_length_computed_mm": round(fl_computed, 2),
            "dist_coeffs":              dc.tolist(),
            "camera_matrix":            cam_mtx.tolist(),
            "optical_center_world":     mean_center.tolist(),
            "used_frames":              len(obj_points),
            "per_image_errors":         pie,
            "confidence":               _confidence(rms),
            "error":                    None,
        })

        # Update K_best tracking after appending result
        valid_so_far = [r for r in fl_results if r.get("rms") is not None]
        if valid_so_far:
            best = min(valid_so_far, key=lambda r: r["rms"])
            if best["focal_length_mm"] == fl_mm:
                K_best = cam_mtx.copy()
                D_best = dist_c.copy()

    # ── Nodal offsets: Z-shift vs the best-calibrated FL ──────────────────
    valid = [(i, c) for i, c in enumerate(optical_centers) if c is not None]
    nodal_offsets: dict[str, float] = {}
    nodal_model_dict: dict | None = None

    if valid:
        best_idx = min(
            valid,
            key=lambda v: fl_results[v[0]].get("rms") or float("inf"),
        )[0]
        ref_z = float(optical_centers[best_idx][2])
        for idx, center in valid:
            fl_mm_val = fl_results[idx]["focal_length_mm"]
            key = str(int(fl_mm_val)) if fl_mm_val == int(fl_mm_val) else f"{fl_mm_val:.1f}"
            nodal_offsets[key] = round(float(center[2]) - ref_z, 2)

        # Fit Padé/poly model for extrapolation beyond calibrated range
        if len(valid) >= 2:
            try:
                mfl = np.array([fl_results[i]["focal_length_mm"] for i, _ in valid])
                mnz = np.array([nodal_offsets[
                    str(int(fl_results[i]["focal_length_mm"]))
                    if fl_results[i]["focal_length_mm"] == int(fl_results[i]["focal_length_mm"])
                    else f"{fl_results[i]['focal_length_mm']:.1f}"
                ] for i, _ in valid])
                nodal_model_dict = fit_nodal_model(mfl, mnz)
            except Exception:
                nodal_model_dict = None

    fl_interpolated = _interpolate_fl_table(
        fl_results, nodal_offsets, fl_groups, nodal_model_dict
    )

    return {
        "success":           True,
        "error":             None,
        "fl_results":        fl_results,
        "fl_interpolated":   fl_interpolated,
        "nodal_offsets_mm":  nodal_offsets,
        "nodal_model":       nodal_model_dict,
        "squeeze_ratio":     squeeze_ratio,
        "lens_type":         "anamorphic" if squeeze_ratio > 1.0 else "spherical",
    }


def _pose_only_calibration(
    usable: list,
    K_fixed: "np.ndarray | None",
    D_fixed: "np.ndarray | None",
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    squeeze_ratio: float,
    working_dist_mm: float,
    fl_mm: float,
) -> dict:
    """
    Estimate the optical centre using solvePnP with a fixed camera matrix.

    Used as a fallback for focal-length groups where not enough full-board
    frames are available for a standalone calibrateCamera call (e.g. at long
    focal lengths where the chart partially overflows the frame).

    Returns a fl_result dict.  The '_mean_center' key carries the corrected
    optical-centre np.ndarray for nodal-offset computation (stripped before
    the dict is returned to the caller).
    """
    if K_fixed is None:
        return {
            "focal_length_mm": fl_mm,
            "rms": None,
            "error": "No reference K available for pose-only calibration",
        }

    D_ref = D_fixed if D_fixed is not None else np.zeros((5, 1), dtype=np.float64)

    centers = []
    for frame in usable:
        corners = np.array(frame["corners"], dtype=np.float32).reshape(-1, 1, 2)
        partial_size = frame.get("partial_grid_size")
        if partial_size:
            p_cols, p_rows = int(partial_size[0]), int(partial_size[1])
            if corners.shape[0] != p_cols * p_rows:
                continue
            obj = _make_objp(p_cols, p_rows, square_size_mm)
        else:
            if corners.shape[0] != board_rows * board_cols:
                continue
            obj = _make_objp(board_cols, board_rows, square_size_mm)

        if squeeze_ratio > 1.0:
            corners = corners.copy()
            corners[:, :, 0] *= squeeze_ratio

        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, corners, K_fixed, D_ref,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            centers.append((-R.T @ tvec).flatten())
        except Exception:
            continue

    if len(centers) < 3:
        return {
            "focal_length_mm": fl_mm,
            "rms": None,
            "error": f"pose-only: only {len(centers)} valid poses from {len(usable)} frames",
        }

    mean_center = np.mean(centers, axis=0).copy()
    if working_dist_mm > 0:
        mean_center[2] -= working_dist_mm

    fx = float(K_fixed[0, 0])
    fy = float(K_fixed[1, 1])
    cx = float(K_fixed[0, 2])
    cy = float(K_fixed[1, 2])

    result = {
        "focal_length_mm":          fl_mm,
        "rms":                      None,   # no full calibration
        "fx_px":                    round(fx, 2),
        "fy_px":                    round(fy, 2),
        "cx_px":                    round(cx, 2),
        "cy_px":                    round(cy, 2),
        "focal_length_computed_mm": 0.0,
        "dist_coeffs":              D_ref.flatten().tolist(),
        "camera_matrix":            K_fixed.tolist(),
        "optical_center_world":     mean_center.tolist(),
        "used_frames":              len(centers),
        "per_image_errors":         [],
        "confidence":               "marginal",
        "error":                    None,
        "pose_only":                True,
        "_mean_center":             mean_center,  # stripped by caller
    }
    return result


def _interpolate_fl_table(
    fl_results: list[dict],
    nodal_offsets: dict[str, float],
    fl_groups: list[dict] | None = None,
    nodal_model_dict: dict | None = None,
) -> list[dict]:
    """
    Build a dense, 1 mm-step interpolated table of lens parameters.

    Uses PCHIP for interpolation between measured points.  When a
    nodal_model_dict is provided, extends the table via the Padé/poly model
    beyond the calibrated FL range up to the outermost *requested* FL
    (determined from fl_groups).
    """
    valid = sorted(
        [r for r in fl_results if r.get("rms") is not None],
        key=lambda r: r["focal_length_mm"],
    )

    # Pose-only results (rms=None but optical_center_world present) can also
    # contribute intrinsic parameters via extrapolation from the Padé model;
    # include them in the PCHIP curve if they have fx_px etc.
    pose_only = sorted(
        [r for r in fl_results if r.get("rms") is None and r.get("fx_px") is not None],
        key=lambda r: r["focal_length_mm"],
    )

    if len(valid) < 2:
        return []

    fls = np.array([r["focal_length_mm"] for r in valid])
    fl_min, fl_max = fls[0], fls[-1]
    if fl_max - fl_min < 2.0:
        return []

    # Determine the full requested FL range (for extrapolation)
    requested_fl_max = fl_max
    if fl_groups:
        all_requested = [float(g["focal_length_mm"]) for g in fl_groups]
        if all_requested:
            requested_fl_max = max(requested_fl_max, max(all_requested))
    requested_fl_min = fl_min
    if fl_groups:
        all_requested = [float(g["focal_length_mm"]) for g in fl_groups]
        if all_requested:
            requested_fl_min = min(requested_fl_min, min(all_requested))

    # Parameters to interpolate
    param_keys = ["fx_px", "fy_px", "cx_px", "cy_px"]
    dc_len = max(len(r.get("dist_coeffs", [])) for r in valid)

    param_data: dict[str, np.ndarray] = {}
    for k in param_keys:
        param_data[k] = np.array([r[k] for r in valid])
    for di in range(dc_len):
        param_data[f"dc_{di}"] = np.array([
            r["dist_coeffs"][di] if di < len(r.get("dist_coeffs", [])) else 0.0
            for r in valid
        ])

    def _nz_key(fl: float) -> str:
        return str(int(fl)) if fl == int(fl) else f"{fl:.1f}"

    nz_vals = np.array([nodal_offsets.get(_nz_key(r["focal_length_mm"]), 0.0) for r in valid])

    # PCHIP interpolators for intrinsics
    interp: dict[str, PchipInterpolator] = {
        k: PchipInterpolator(fls, v) for k, v in param_data.items()
    }
    interp["nz"] = PchipInterpolator(fls, nz_vals)

    # Build the FL query set: dense points within calibrated range + extrapolated range
    calibrated_set = set(fls.tolist())
    step = max(1.0, (requested_fl_max - requested_fl_min) / 200)
    query_fls = np.arange(requested_fl_min, requested_fl_max + step * 0.5, step)

    rows: list[dict] = []
    for fl in query_fls:
        fl = round(float(fl), 4)
        if any(abs(fl - cf) < 0.01 for cf in calibrated_set):
            continue

        # Intrinsics: PCHIP within calibrated range, linear extrapolation outside
        cam_fx = float(interp["fx_px"](fl))
        cam_fy = float(interp["fy_px"](fl))
        cam_cx = float(interp["cx_px"](fl))
        cam_cy = float(interp["cy_px"](fl))
        dc     = [float(interp[f"dc_{di}"](fl)) for di in range(dc_len)]

        # Nodal offset: PCHIP inside calibrated range, Padé model outside
        if fl_min <= fl <= fl_max:
            nz = round(float(interp["nz"](fl)), 4)
            extrapolated = False
        elif nodal_model_dict is not None:
            nz = round(float(predict_nodal(nodal_model_dict, np.array([fl]))[0]), 4)
            extrapolated = True
        else:
            # Linear extension from boundary
            nz = round(float(interp["nz"](fl)), 4)
            extrapolated = True

        cam_mtx = [
            [cam_fx, 0.0, cam_cx],
            [0.0, cam_fy, cam_cy],
            [0.0, 0.0, 1.0],
        ]
        rows.append({
            "focal_length_mm":   fl,
            "rms":               None,
            "fx_px":             round(cam_fx, 2),
            "fy_px":             round(cam_fy, 2),
            "cx_px":             round(cam_cx, 2),
            "cy_px":             round(cam_cy, 2),
            "dist_coeffs":       [round(v, 8) for v in dc],
            "camera_matrix":     cam_mtx,
            "nodal_offset_z_mm": nz,
            "interpolated":      True,
            "extrapolated":      extrapolated,
        })

    return rows


def _make_objp(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """Build a planar (z=0) object-point array for a cols×rows inner-corner grid."""
    pts = np.zeros((rows * cols, 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return pts


def _confidence(rms: float) -> str:
    if rms < 0.3: return "excellent"
    if rms < 0.5: return "good"
    if rms < 1.0: return "marginal"
    return "poor"
