"""
Zoom sweep calibration: calibrate at multiple focal lengths and estimate
per-FL nodal offsets from optical-center geometry.

For accurate nodal offsets keep the camera at a similar working distance
for all focal lengths — only change the zoom ring.
"""
from __future__ import annotations

import numpy as np
import cv2


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
        nodal_offsets_mm: {str(fl_mm): float, ...},   # Z-shift vs first valid FL
    }
    """
    objp = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    objp[:, :2] = (
        np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_mm
    )

    fl_results: list[dict] = []
    optical_centers: list          = []   # None or np.ndarray per group

    for group in fl_groups:
        fl_mm   = float(group["focal_length_mm"])
        frames  = group.get("frames", [])
        usable  = [f for f in frames
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
            if corners.shape[0] != board_rows * board_cols:
                continue
            if squeeze_ratio > 1.0:
                corners = corners.copy()
                corners[:, :, 0] *= squeeze_ratio  # scale x to de-squeezed space
            obj_points.append(objp)
            img_points.append(corners)
            paths.append(frame.get("path", ""))

        if len(obj_points) < 3:
            fl_results.append({
                "focal_length_mm": fl_mm, "rms": None,
                "error": "Not enough corners after filtering",
            })
            optical_centers.append(None)
            continue

        calib_size = (int(image_size[0] * squeeze_ratio), image_size[1]) if squeeze_ratio > 1.0 else image_size
        try:
            rms, cam_mtx, dist_c, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, calib_size, None, None
            )
        except Exception as exc:
            fl_results.append({"focal_length_mm": fl_mm, "rms": None, "error": str(exc)})
            optical_centers.append(None)
            continue

        fx_px = float(cam_mtx[0, 0])
        fy_px = float(cam_mtx[1, 1])
        cx_px = float(cam_mtx[0, 2])
        cy_px = float(cam_mtx[1, 2])
        dc    = dist_c.flatten()

        fl_computed = (fx_px * sensor_width_mm / calib_size[0]
                       if sensor_width_mm > 0 else 0.0)

        # ── Optical centre for each image ──────────────────────────────
        centers = []
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            centers.append((-R.T @ tv).flatten())
        mean_center = np.mean(centers, axis=0)
        optical_centers.append(mean_center)

        # ── Per-image reprojection errors ──────────────────────────────
        pie = []
        for i, (obj, img, rv, tv) in enumerate(
                zip(obj_points, img_points, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(obj, rv, tv, cam_mtx, dist_c)
            err = float(np.sqrt(np.mean(
                np.sum((img.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1)
            )))
            pie.append({"path": paths[i], "error": round(err, 4), "outlier": False})

        # IQR-based outlier detection (robust to skewed distributions)
        pie_arr = np.array([e["error"] for e in pie])
        q1, q3 = float(np.percentile(pie_arr, 25)), float(np.percentile(pie_arr, 75))
        iqr = q3 - q1
        pie_threshold = q3 + 1.5 * iqr if iqr > 0 else float(np.median(pie_arr)) * 2.0
        for entry in pie:
            entry["outlier"] = entry["error"] > pie_threshold

        fl_results.append({
            "focal_length_mm":         fl_mm,
            "rms":                     round(rms, 4),
            "fx_px":                   round(fx_px, 2),
            "fy_px":                   round(fy_px, 2),
            "cx_px":                   round(cx_px, 2),
            "cy_px":                   round(cy_px, 2),
            "focal_length_computed_mm": round(fl_computed, 2),
            "dist_coeffs":             dc.tolist(),
            "camera_matrix":           cam_mtx.tolist(),
            "optical_center_world":    mean_center.tolist(),
            "used_frames":             len(obj_points),
            "per_image_errors":        pie,
            "confidence":              _confidence(rms),
            "error":                   None,
        })

    # ── Nodal offsets: Z-shift vs the best-calibrated FL ──────────────
    # Use the FL with the lowest RMS as the reference to minimise error
    # propagation. Keys are formatted as integers or ".1f" floats so they
    # match the lookup in export_ue5_ulens_zoom (which also uses str(fl_mm)).
    valid = [(i, c) for i, c in enumerate(optical_centers) if c is not None]
    nodal_offsets: dict[str, float] = {}
    if valid:
        # Pick the index with the best (lowest) RMS among valid calibrations
        best_idx = min(
            valid,
            key=lambda v: fl_results[v[0]].get("rms") or float("inf"),
        )[0]
        ref_z = float(optical_centers[best_idx][2])
        for idx, center in valid:
            fl_mm = fl_results[idx]["focal_length_mm"]
            key = str(int(fl_mm)) if fl_mm == int(fl_mm) else f"{fl_mm:.1f}"
            nodal_offsets[key] = round(float(center[2]) - ref_z, 2)

    return {
        "success":         True,
        "error":           None,
        "fl_results":      fl_results,
        "nodal_offsets_mm": nodal_offsets,
        "squeeze_ratio":   squeeze_ratio,
        "lens_type":       "anamorphic" if squeeze_ratio > 1.0 else "spherical",
    }


def _confidence(rms: float) -> str:
    if rms < 0.3: return "excellent"
    if rms < 0.5: return "good"
    if rms < 1.0: return "marginal"
    return "poor"
