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

Interpolation uses physics-informed models that match actual lens optics:
  - fx/fy (pixels)     : linear in fl_mm  (optics law: f_px ∝ f_mm)
  - k1/k2/k3           : a/fl² + b        (radial distortion ∝ 1/f²)
  - cx/cy/p1/p2        : linear           (principal-point drift / zoom breathing)
  - nodal_offset_z_mm  : linear

These models extrapolate and interpolate more accurately than general-purpose
splines when data is sparse (2–3 control points), because they are constrained
to the shapes that real lens behaviour actually follows.
"""
from __future__ import annotations

import numpy as np
import cv2
from collections import Counter
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Physics-informed curve models
# ---------------------------------------------------------------------------

def _linear(x, a, b):
    """Linear: a·x + b  (best model for fx/fy vs fl, and cx/cy drift)."""
    return a * x + b


def _inv_sq(x, a, b):
    """Hyperbolic: a/x² + b  (radial distortion k1/k2/k3 vs fl)."""
    return a / (x * x) + b


def _fit_or_pchip(fls: np.ndarray, values: np.ndarray, model, query: np.ndarray,
                  fallback_pchip: PchipInterpolator | None = None) -> np.ndarray:
    """
    Try to fit *model* via curve_fit. If it converges and residuals are
    reasonable, use it; otherwise fall back to PCHIP.
    """
    try:
        popt, _ = curve_fit(model, fls, values, maxfev=4000)
        fitted = model(query, *popt)
        # Sanity-check residuals on the known points
        residuals = np.abs(model(fls, *popt) - values)
        if np.max(residuals) < max(np.std(values) * 0.15 + 1e-9, 0.5):
            return fitted
    except Exception:
        pass
    # Fall back to PCHIP
    if fallback_pchip is None:
        fallback_pchip = PchipInterpolator(fls, values)
    return fallback_pchip(query)

from nodal_model import fit_nodal_model, predict_nodal


def _estimate_scale_mm(points_xyz: np.ndarray) -> float:
    """
    Estimate a robust spatial scale for a set of object points.

    Using a normalized point cloud improves PnP/calibration conditioning when
    chart dimensions are very large, while keeping the recovered translation in
    physical millimetres after de-normalization.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return 1.0

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    if np.isfinite(diag) and diag > 1e-6:
        return diag

    radius = float(np.max(np.linalg.norm(pts, axis=1)))
    if np.isfinite(radius) and radius > 1e-6:
        return radius

    return 1.0


def _estimate_group_scale_mm(obj_points: list[np.ndarray]) -> float:
    """Estimate one normalization scale for an FL group of object-point sets."""
    if not obj_points:
        return 1.0
    merged = np.concatenate([np.asarray(o, dtype=np.float64).reshape(-1, 3) for o in obj_points], axis=0)
    return _estimate_scale_mm(merged)


def _camera_center_from_rt(rvec: np.ndarray, tvec: np.ndarray, scale_mm: float = 1.0) -> np.ndarray:
    """Convert (rvec, tvec) to camera center in world coordinates and de-normalize scale."""
    R, _ = cv2.Rodrigues(rvec)
    center = (-R.T @ tvec).flatten()
    return np.asarray(center, dtype=np.float64) * float(scale_mm)


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
    pending_nodal: list[dict]      = []   # FLs needing PnP-only nodal estimation

    def _is_implausible_solution(cam_mtx: np.ndarray, dist_c: np.ndarray, calib_size_local: tuple) -> bool:
        w, h = calib_size_local
        fx = float(cam_mtx[0, 0])
        fy = float(cam_mtx[1, 1])
        cx = float(cam_mtx[0, 2])
        cy = float(cam_mtx[1, 2])
        dc = dist_c.flatten()
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
        if abs(cx) > 3.0 * w or abs(cy) > 3.0 * h:
            return True
        if abs(k1) > 2.0 or abs(k2) > 2.0:
            return True
        if abs(p1) > 0.2 or abs(p2) > 0.2:
            return True
        return False

    def _run_constrained_fallback(obj_points_local: list, img_points_local: list, calib_size_local: tuple):
        w, h = calib_size_local
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
        return cv2.calibrateCamera(
            obj_points_local, img_points_local, calib_size_local, cam0, dc0, flags=flags
        )

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

        # Keep one detection model per FL solve to avoid mixed board geometry.
        det_counter = Counter((f.get("detection_type") or "checkerboard") for f in usable)
        selected_det = None
        for det_name in ("checkerboard", "aruco_grid", "charuco"):
            if det_counter.get(det_name, 0) >= 3:
                selected_det = det_name
                break
        if selected_det is None:
            selected_det = det_counter.most_common(1)[0][0]
        usable = [f for f in usable if (f.get("detection_type") or "checkerboard") == selected_det]

        obj_points, img_points, paths = [], [], []
        sparse_mode = False
        for frame in usable:
            corners = np.array(frame["corners"], dtype=np.float32).reshape(-1, 1, 2)
            frame_obj_points = frame.get("obj_points")
            partial_size = frame.get("partial_grid_size")
            if frame_obj_points:
                obj = np.array(frame_obj_points, dtype=np.float32).reshape(-1, 3)
                if obj.shape[0] != corners.shape[0] or obj.shape[0] < 6:
                    continue
                if (frame.get("detection_type") or "") == "aruco_grid":
                    obj = obj * float(square_size_mm)
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
                corners[:, :, 0] *= squeeze_ratio
            obj_points.append(obj)
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
        # Some OpenCV builds do not expose CALIB_FIX_SKEW; default to no extra
        # flags in that case so zoom calibration still runs.
        calib_flags = getattr(cv2, "CALIB_FIX_SKEW", 0)
        cam0 = None
        dc0 = None
        if sparse_mode:
            w0, h0 = calib_size
            cam0 = np.array(
                [[max(w0, h0), 0.0, w0 / 2.0],
                 [0.0, max(w0, h0), h0 / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dc0 = np.zeros((5, 1), dtype=np.float64)
            calib_flags |= (
                cv2.CALIB_USE_INTRINSIC_GUESS
                | cv2.CALIB_FIX_K3
                | cv2.CALIB_ZERO_TANGENT_DIST
            )
        group_scale_mm = _estimate_group_scale_mm(obj_points)
        obj_points_norm = [np.asarray(obj, dtype=np.float32) / float(group_scale_mm) for obj in obj_points]

        try:
            rms, cam_mtx, dist_c, rvecs, tvecs = cv2.calibrateCamera(
                obj_points_norm, img_points, calib_size, cam0, dc0, flags=calib_flags
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

        solve_warning = None
        if _is_implausible_solution(cam_mtx, dist_c, calib_size):
            try:
                rms_fb, cam_fb, dist_fb, rv_fb, tv_fb = _run_constrained_fallback(
                    obj_points_norm, img_points, calib_size
                )
                if _is_implausible_solution(cam_fb, dist_fb, calib_size):
                    fl_results.append({
                        "focal_length_mm": fl_mm,
                        "rms": None,
                        "error": "Implausible calibration solution for this FL. Capture more varied poses/frames.",
                    })
                    optical_centers.append(None)
                    continue
                rms, cam_mtx, dist_c, rvecs, tvecs = rms_fb, cam_fb, dist_fb, rv_fb, tv_fb
                solve_warning = "Used constrained tele fallback solver (k2/k3 fixed, tangential fixed)."
            except Exception:
                # Constrained fallback also failed — use pose-only with best known K.
                pose_result = _pose_only_calibration(
                    usable, K_best, D_best, board_cols, board_rows,
                    square_size_mm, squeeze_ratio, working_dist_mm, fl_mm,
                )
                pose_result.setdefault(
                    "error",
                    "Implausible calibration solution for this FL. Capture more varied poses/frames.",
                )
                fl_results.append(pose_result)
                optical_centers.append(pose_result.get("_mean_center"))
                continue

        if not np.isfinite(rms):
            fl_results.append({
                "focal_length_mm": fl_mm,
                "rms": None,
                "error": f"RMS too high ({float(rms):.2f}px) for this FL. Likely board/profile mismatch.",
            })
            pending_nodal.append({
                "index": len(fl_results) - 1,
                "obj_points": obj_points,
                "img_points": img_points,
                "paths": paths,
                "focal_length_mm": fl_mm,
                "detection_type": selected_det,
            })
            optical_centers.append(None)
            continue

        if rms > 5.0:
            if sparse_mode:
                relax_msg = (
                    f"High RMS ({float(rms):.2f}px) in sparse tele mode. "
                    "Kept low-confidence fit for continuity; distortion is likely unreliable."
                )
                solve_warning = f"{solve_warning} {relax_msg}" if solve_warning else relax_msg
            else:
                fl_results.append({
                    "focal_length_mm": fl_mm,
                    "rms": None,
                    "error": f"RMS too high ({float(rms):.2f}px) for this FL. Likely board/profile mismatch.",
                })
                pending_nodal.append({
                    "index": len(fl_results) - 1,
                    "obj_points": obj_points,
                    "img_points": img_points,
                    "paths": paths,
                    "focal_length_mm": fl_mm,
                    "detection_type": selected_det,
                })
                optical_centers.append(None)
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
            centers.append(_camera_center_from_rt(rv, tv, group_scale_mm))
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
                zip(obj_points_norm, img_points, rvecs, tvecs)):
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
            "detection_type":           selected_det,
            "warning":                  solve_warning,
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

    # If some FLs failed distortion solve, still estimate their nodal centre
    # using the best solved FL intrinsics/distortion (PnP-only fallback).
    if valid and pending_nodal:
        ref_idx = min(
            valid,
            key=lambda v: fl_results[v[0]].get("rms") or float("inf"),
        )[0]
        ref_cam = np.array(fl_results[ref_idx]["camera_matrix"], dtype=np.float64)
        ref_dc = np.array(fl_results[ref_idx]["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        for item in pending_nodal:
            centers = []
            pie = []
            for obj, img, path in zip(item["obj_points"], item["img_points"], item["paths"]):
                obj_pnp = np.array(obj, dtype=np.float32).reshape(-1, 3)
                img_pnp = np.array(img, dtype=np.float32).reshape(-1, 1, 2)
                if obj_pnp.shape[0] < 6:
                    continue
                pnp_scale_mm = _estimate_scale_mm(obj_pnp)
                obj_pnp_norm = obj_pnp / float(pnp_scale_mm)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pnp_norm,
                    img_pnp,
                    ref_cam,
                    ref_dc,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    continue
                centers.append(_camera_center_from_rt(rvec, tvec, pnp_scale_mm))
                proj, _ = cv2.projectPoints(obj_pnp_norm, rvec, tvec, ref_cam, ref_dc)
                err = float(np.sqrt(np.mean(
                    np.sum((img_pnp.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1)
                )))
                pie.append({"path": path, "error": round(err, 4), "outlier": False})

            if len(centers) < 3:
                continue

            mean_center = np.mean(np.array(centers), axis=0)
            idx = int(item["index"])
            optical_centers[idx] = mean_center
            fl_results[idx].update({
                "camera_matrix": ref_cam.tolist(),
                "dist_coeffs": ref_dc.flatten().tolist(),
                "fx_px": round(float(ref_cam[0, 0]), 2),
                "fy_px": round(float(ref_cam[1, 1]), 2),
                "cx_px": round(float(ref_cam[0, 2]), 2),
                "cy_px": round(float(ref_cam[1, 2]), 2),
                "optical_center_world": mean_center.tolist(),
                "used_frames": len(centers),
                "per_image_errors": pie,
                "warning": (
                    "Nodal estimated via PnP using reference FL intrinsics/distortion; "
                    "distortion at this FL not independently solved."
                ),
                "error": None,
            })

    # Recompute after any nodal fallback updates.
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
            pnp_scale_mm = _estimate_scale_mm(obj)
            obj_norm = obj / float(pnp_scale_mm)
            ok, rvec, tvec = cv2.solvePnP(
                obj_norm, corners, K_fixed, D_ref,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            centers.append(_camera_center_from_rt(rvec, tvec, pnp_scale_mm))
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
    Build a dense, interpolated table of lens parameters using physics-informed
    curve models that match real lens optics behaviour.

    Models used
    -----------
    fx_px, fy_px     linear(fl)          f_px ∝ f_mm  (thin-lens law)
    cx_px, cy_px     linear(fl)          zoom breathing / principal-point drift
    k1, k2, k3       a/fl² + b           radial distortion ∝ 1/f²
    p1, p2           linear(fl)          tangential coefficients are small & stable
    nodal_offset_z   linear(fl)

    Falls back to monotonic-cubic PCHIP for any parameter where the physics
    model fit residuals are too large (e.g. unusual optical design).

    When a nodal_model_dict is provided, extends the table via the Padé/poly
    model beyond the calibrated FL range up to the outermost *requested* FL
    (determined from fl_groups).

    Requires ≥ 2 successfully calibrated focal lengths.
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

    fls = np.array([r["focal_length_mm"] for r in valid], dtype=float)
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

    # ── Build query grid across the full requested range (≤ 200 steps) ──
    calibrated_set = set(fls.tolist())
    step = max(1.0, (requested_fl_max - requested_fl_min) / 200)
    query_fls = np.arange(requested_fl_min, requested_fl_max + step * 0.5, step)

    # ── Gather per-parameter arrays ──────────────────────────────────────
    def _nz_key(fl_mm: float) -> str:
        return str(int(fl_mm)) if fl_mm == int(fl_mm) else f"{fl_mm:.1f}"

    fx_vals  = np.array([r["fx_px"] for r in valid], dtype=float)
    fy_vals  = np.array([r["fy_px"] for r in valid], dtype=float)
    cx_vals  = np.array([r["cx_px"] for r in valid], dtype=float)
    cy_vals  = np.array([r["cy_px"] for r in valid], dtype=float)
    nz_vals  = np.array([nodal_offsets.get(_nz_key(r["focal_length_mm"]), 0.0) for r in valid], dtype=float)

    dc_len = max(len(r.get("dist_coeffs", [])) for r in valid)
    dc_arrays = []
    for di in range(dc_len):
        dc_arrays.append(np.array([
            r["dist_coeffs"][di] if di < len(r.get("dist_coeffs", [])) else 0.0
            for r in valid
        ], dtype=float))

    # ── Build PCHIP fallbacks (works for any ≥2 points) ─────────────────
    pchip_fx  = PchipInterpolator(fls, fx_vals)
    pchip_fy  = PchipInterpolator(fls, fy_vals)
    pchip_cx  = PchipInterpolator(fls, cx_vals)
    pchip_cy  = PchipInterpolator(fls, cy_vals)
    pchip_nz  = PchipInterpolator(fls, nz_vals)
    pchip_dc  = [PchipInterpolator(fls, dc_arrays[di]) for di in range(dc_len)]

    # Physics model only makes sense with ≥2 points (which we already guarantee)
    q = query_fls
    fx_interp  = _fit_or_pchip(fls, fx_vals,  _linear,  q, pchip_fx)
    fy_interp  = _fit_or_pchip(fls, fy_vals,  _linear,  q, pchip_fy)
    cx_interp  = _fit_or_pchip(fls, cx_vals,  _linear,  q, pchip_cx)
    cy_interp  = _fit_or_pchip(fls, cy_vals,  _linear,  q, pchip_cy)
    nz_interp  = _fit_or_pchip(fls, nz_vals,  _linear,  q, pchip_nz)

    # Distortion: k1/k2/k3 (indices 0,1,4) use inv-sq model; p1/p2 (2,3) use linear
    dc_interps = []
    for di in range(dc_len):
        model = _inv_sq if di in (0, 1, 4) else _linear
        dc_interps.append(_fit_or_pchip(fls, dc_arrays[di], model, q, pchip_dc[di]))

    # ── Build output rows ─────────────────────────────────────────────────
    rows: list[dict] = []
    for qi, fl in enumerate(query_fls):
        fl = round(float(fl), 4)
        if any(abs(fl - cf) < 0.01 for cf in calibrated_set):
            continue  # skip rows that match a measured point

        cam_fx = float(fx_interp[qi])
        cam_fy = float(fy_interp[qi])
        cam_cx = float(cx_interp[qi])
        cam_cy = float(cy_interp[qi])
        dc     = [float(dc_interps[di][qi]) for di in range(dc_len)]

        # Nodal offset: physics-informed interpolation inside calibrated range,
        # Padé model outside (telephoto extrapolation).
        if fl_min <= fl <= fl_max:
            nz = round(float(nz_interp[qi]), 4)
            extrapolated = False
        elif nodal_model_dict is not None:
            nz = round(float(predict_nodal(nodal_model_dict, np.array([fl]))[0]), 4)
            extrapolated = True
        else:
            # Linear extension from boundary via PCHIP extrapolation
            nz = round(float(nz_interp[qi]), 4)
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
