"""
Calibration result exporters.

Supported formats:
  - OpenCV FileStorage XML  (Ventuz FreeD compatible)
  - JSON                    (generic, with metadata)
  - STmap EXR               (32-bit float, UV remap, requires openexr + imath)
  - UE5 .ulens              (CameraCalibration plugin SphericalLensModel)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

# OpenEXR is optional — export_stmap_exr will return an error if missing
try:
    import OpenEXR
    _HAS_EXR = True
except ImportError:
    _HAS_EXR = False


# ---------------------------------------------------------------------------
# 1.  OpenCV FileStorage XML
# ---------------------------------------------------------------------------

def export_opencv_xml(
    path: str,
    camera_matrix: list,
    dist_coeffs: list,
    fov_x: float,
    fov_y: float,
    image_size: tuple,  # (width, height)
) -> dict:
    try:
        cm = np.array(camera_matrix, dtype=np.float64)
        dc = np.array(dist_coeffs, dtype=np.float64).flatten()
        w, h = int(image_size[0]), int(image_size[1])

        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write("imageWidth", w)
        fs.write("imageHeight", h)
        fs.write("camMtx", cm)
        fs.write("distCoeffs", dc.reshape(1, -1))
        fs.write("fovX", float(fov_x))
        fs.write("fovY", float(fov_y))
        fs.release()

        return {"success": True, "output_path": os.path.abspath(path), "error": None}
    except Exception as exc:
        return {"success": False, "output_path": path, "error": str(exc)}


# ---------------------------------------------------------------------------
# 2.  JSON
# ---------------------------------------------------------------------------

def export_json(
    path: str,
    camera_matrix: list,
    dist_coeffs: list,
    fov_x: float,
    fov_y: float,
    rms: float,
    metadata: Optional[dict] = None,
    squeeze_ratio: float = 1.0,
    lens_type: str = "spherical",
) -> dict:
    try:
        cm = np.array(camera_matrix, dtype=np.float64)
        dc = np.array(dist_coeffs, dtype=np.float64).flatten()

        payload = {
            "calibration_date": datetime.now(timezone.utc).isoformat(),
            "rms": rms,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "lens_type": lens_type,
            "squeeze_ratio": round(squeeze_ratio, 4),
            "camera_matrix": {
                "fx": cm[0, 0],
                "fy": cm[1, 1],
                "cx": cm[0, 2],
                "cy": cm[1, 2],
                "rows": cm.tolist(),
            },
            "dist_coeffs": dc.tolist(),
            **(metadata or {}),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return {"success": True, "output_path": os.path.abspath(path), "error": None}
    except Exception as exc:
        return {"success": False, "output_path": path, "error": str(exc)}


# ---------------------------------------------------------------------------
# 3.  STmap EXR
# ---------------------------------------------------------------------------

def export_stmap_exr(
    path: str,
    camera_matrix: list,
    dist_coeffs: list,
    image_size: tuple,  # (width, height)
    squeeze_ratio: float = 1.0,
) -> dict:
    if not _HAS_EXR:
        return {
            "success": False,
            "output_path": path,
            "error": "OpenEXR and Imath are not installed. Run: pip install openexr imath",
        }

    try:
        cm = np.array(camera_matrix, dtype=np.float64)
        dc = np.array(dist_coeffs, dtype=np.float64)
        w, h = int(image_size[0]), int(image_size[1])
        w_out = int(w * squeeze_ratio) if squeeze_ratio > 1.0 else w

        map_x, map_y = cv2.initUndistortRectifyMap(
            cm, dc, None, cm, (w_out, h), cv2.CV_32FC1
        )

        # Normalize to 0–1 UV space (STmap convention).
        # map_x is in de-squeezed output pixel space (w_out wide); normalize
        # across the full output width so U spans exactly [0, 1].
        u = (map_x / max(w_out - 1, 1)).astype(np.float32)
        v = (map_y / (h - 1)).astype(np.float32)
        zeros = np.zeros((h, w_out), dtype=np.float32)

        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "dataWindow": ((0, 0), (w_out - 1, h - 1)),
            "displayWindow": ((0, 0), (w_out - 1, h - 1)),
        }
        channels = {
            "R": OpenEXR.Channel(u),
            "G": OpenEXR.Channel(v),
            "B": OpenEXR.Channel(zeros),
        }
        part = OpenEXR.Part(header, channels)
        OpenEXR.File([part]).write(path)

        return {"success": True, "output_path": os.path.abspath(path), "error": None}
    except Exception as exc:
        return {"success": False, "output_path": path, "error": str(exc)}


# ---------------------------------------------------------------------------
# 4.  UE5 .ulens (CameraCalibration plugin — SphericalLensModel)
# ---------------------------------------------------------------------------

def _ulens_row(*values: float) -> str:
    """Format one data row as comma-separated floats."""
    return ", ".join(f"{v:.6g}" for v in values)


def _ulens_data(rows: list) -> str:
    """Encode a list of rows as a semicolon-separated data string."""
    return "; ".join(_ulens_row(*row) for row in rows)


def export_ue5_ulens(
    path: str,
    camera_matrix: list,
    dist_coeffs: list,
    fov_x: float,           # not used in file body; kept for API consistency
    fov_y: float,           # not used in file body; kept for API consistency
    image_size: tuple,      # (width, height) in pixels
    lens_name: str = "Lens",
    sensor_width_mm: float = 0.0,
    sensor_height_mm: float = 0.0,
    squeeze_ratio: float = 1.0,
    lens_type: str = "spherical",
) -> dict:
    """
    Produces a .ulens JSON file matching the format used by UE5's
    CameraCalibration plugin (as exported by e.g. Pomfort Silverstack /
    Teradek RT / Mosys).

    Normalisation:
      Fx = camera_matrix[0,0] / image_width
      Fy = camera_matrix[1,1] / image_height
      Cx = camera_matrix[0,2] / image_width
      Cy = camera_matrix[1,2] / image_height

    Data strings use semicolons between rows and commas between values.
    """
    try:
        cm = np.array(camera_matrix, dtype=np.float64)
        dc = np.array(dist_coeffs, dtype=np.float64).flatten()
        w, h = int(image_size[0]), int(image_size[1])
        calib_w = int(w * squeeze_ratio) if squeeze_ratio > 1.0 else w

        fx = float(cm[0, 0]) / calib_w
        fy = float(cm[1, 1]) / h
        cx = float(cm[0, 2]) / calib_w
        cy = float(cm[1, 2]) / h

        # OpenCV dist_coeffs order: k1 k2 p1 p2 [k3 ...]
        def _dc(i: int) -> float:
            return float(dc[i]) if len(dc) > i else 0.0

        k1, k2, p1, p2, k3 = _dc(0), _dc(1), _dc(2), _dc(3), _dc(4)

        # Static lens: single entry at focus=0.0, zoom=0.0
        focus_enc = 0.0
        zoom_enc  = 0.0

        lens_info: dict = {
            "serialNumber": "None",
            "modelName": lens_name,
            "distortionModel": "Spherical",
        }
        if squeeze_ratio > 1.0:
            lens_info["squeezeRatio"] = round(squeeze_ratio, 4)

        user_metadata = []
        if squeeze_ratio > 1.0:
            user_metadata.append({"key": "lensType", "value": "anamorphic"})
            user_metadata.append({"key": "squeezeRatio", "value": str(round(squeeze_ratio, 4))})

        ulens = {
            "metadata": {
                "type": "LensFile",
                "version": "0.0.0",
                "lensInfo": lens_info,
                "name": lens_name,
                "nodalOffsetCoordinateSystem": "OpenCV",
                "fxFyUnits": "Normalized",
                "cxCyUnits": "Normalized",
                "userMetadata": user_metadata,
            },
            "sensorDimensions": {
                "width":  round(sensor_width_mm,  6),
                "height": round(sensor_height_mm, 6),
                "units":  "Millimeters",
            },
            "imageDimensions": {
                "width":  w,
                "height": h,
            },
            "cameraParameterTables": [
                {
                    "parameterName": "FocalLengthTable",
                    "header": "FocusEncoder, ZoomEncoder, Fx, Fy",
                    "data": _ulens_data([[focus_enc, zoom_enc, fx, fy]]),
                },
                {
                    "parameterName": "ImageCenterTable",
                    "header": "FocusEncoder, ZoomEncoder, Cx, Cy",
                    "data": _ulens_data([[focus_enc, zoom_enc, cx, cy]]),
                },
                {
                    # Identity nodal offset — unknown without tracker data
                    # Column order matches reference files: Qx Qy Qz Qw Tx Ty Tz
                    "parameterName": "NodalOffsetTable",
                    "header": "FocusEncoder, ZoomEncoder, Qx, Qy, Qz, Qw, Tx, Ty, Tz",
                    "data": _ulens_data([[focus_enc, zoom_enc, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                },
                {
                    "parameterName": "DistortionTable",
                    "header": "FocusEncoder, ZoomEncoder, K1, K2, K3, P1, P2",
                    "data": _ulens_data([[focus_enc, zoom_enc, k1, k2, k3, p1, p2]]),
                },
            ],
            "encoderTables": [
                {
                    "parameterName": "Focus",
                    "header": "FocusEncoder, FocusCM",
                    "data": "",
                },
                {
                    "parameterName": "Iris",
                    "header": "IrisEncoder, IrisFstop",
                    "data": "",
                },
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(ulens, f, indent="\t")

        return {"success": True, "output_path": os.path.abspath(path), "error": None}
    except Exception as exc:
        return {"success": False, "output_path": path, "error": str(exc)}


def export_ue5_ulens_zoom(
    path: str,
    fl_results: list,
    image_size: tuple,          # (width, height)
    nodal_offsets_mm: dict,     # {str(fl_mm): float}
    lens_name: str = "Lens",
    sensor_width_mm: float = 0.0,
    sensor_height_mm: float = 0.0,
    squeeze_ratio: float = 1.0,
    lens_type: str = "spherical",
    fl_interpolated: Optional[list] = None,
) -> dict:
    """
    Export a multi-focal-length .ulens file for zoom lenses.

    When ``fl_interpolated`` is supplied (from run_zoom_calibration), the
    dense interpolated rows are merged with the calibrated rows to give UE5
    a fine-grained zoom table, preventing linear-interpolation artefacts
    across large focal-length gaps.

    ZoomEncoder is normalised 0–1 across the captured FL range.
    NodalOffset Tz is the optical-centre Z-shift in mm vs the best-RMS FL.
    """
    try:
        w, h = int(image_size[0]), int(image_size[1])
        calib_w = int(w * squeeze_ratio) if squeeze_ratio > 1.0 else w

        # Calibrated rows (have a real camera_matrix)
        calibrated = sorted(
            [r for r in fl_results if r.get("rms") is not None],
            key=lambda r: r["focal_length_mm"],
        )
        if not calibrated:
            return {"success": False, "output_path": path,
                    "error": "No valid focal-length results to export"}

        fl_min   = calibrated[0]["focal_length_mm"]
        fl_max   = calibrated[-1]["focal_length_mm"]
        fl_range = max(fl_max - fl_min, 1.0)
        focus_enc = 0.0

        def ze(fl_mm: float) -> float:
            return round((fl_mm - fl_min) / fl_range, 6)

        # Merge calibrated + interpolated rows, sorted by FL.
        # Interpolated rows carry pre-computed fx_px/fy_px/cx_px/cy_px and
        # nodal_offset_z_mm directly (no camera_matrix needed).
        all_rows = list(calibrated)
        if fl_interpolated:
            all_rows += fl_interpolated
        all_rows.sort(key=lambda r: r["focal_length_mm"])

        fl_rows, ic_rows, nd_rows, dist_rows = [], [], [], []

        for r in all_rows:
            fl_mm = r["focal_length_mm"]
            is_interp = r.get("interpolated", False)

            if is_interp:
                # Interpolated row: parameters are already in pixel units
                fx_n = r["fx_px"] / calib_w
                fy_n = r["fy_px"] / h
                cx_n = r["cx_px"] / calib_w
                cy_n = r["cy_px"] / h
                dc   = r.get("dist_coeffs", [0.0] * 5)

                def _di(i: int) -> float:
                    return float(dc[i]) if i < len(dc) else 0.0

                k1, k2, p1, p2, k3 = _di(0), _di(1), _di(2), _di(3), _di(4)
                nz = float(r.get("nodal_offset_z_mm", 0.0))
            else:
                cm = np.array(r["camera_matrix"], dtype=np.float64)
                dc_arr = np.array(r["dist_coeffs"], dtype=np.float64).flatten()
                fx_n = float(cm[0, 0]) / calib_w
                fy_n = float(cm[1, 1]) / h
                cx_n = float(cm[0, 2]) / calib_w
                cy_n = float(cm[1, 2]) / h

                def _d(i: int) -> float:
                    return float(dc_arr[i]) if len(dc_arr) > i else 0.0

                k1, k2, p1, p2, k3 = _d(0), _d(1), _d(2), _d(3), _d(4)
                _nz_key = str(int(fl_mm)) if fl_mm == int(fl_mm) else f"{fl_mm:.1f}"
                nz = float(nodal_offsets_mm.get(_nz_key, 0.0))

            fl_rows.append([focus_enc, ze(fl_mm), fx_n, fy_n])
            ic_rows.append([focus_enc, ze(fl_mm), cx_n, cy_n])
            nd_rows.append([focus_enc, ze(fl_mm), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, nz])
            dist_rows.append([focus_enc, ze(fl_mm), k1, k2, k3, p1, p2])

        zoom_lens_info: dict = {
            "serialNumber": "None",
            "modelName":    lens_name,
            "distortionModel": "Spherical",
        }
        if squeeze_ratio > 1.0:
            zoom_lens_info["squeezeRatio"] = round(squeeze_ratio, 4)

        zoom_user_metadata = []
        if squeeze_ratio > 1.0:
            zoom_user_metadata.append({"key": "lensType", "value": "anamorphic"})
            zoom_user_metadata.append({"key": "squeezeRatio", "value": str(round(squeeze_ratio, 4))})

        ulens = {
            "metadata": {
                "type": "LensFile",
                "version": "0.0.0",
                "lensInfo": zoom_lens_info,
                "name":                          lens_name,
                "nodalOffsetCoordinateSystem":   "OpenCV",
                "fxFyUnits":                     "Normalized",
                "cxCyUnits":                     "Normalized",
                "userMetadata":                  zoom_user_metadata,
            },
            "sensorDimensions": {
                "width":  round(sensor_width_mm,  6),
                "height": round(sensor_height_mm, 6),
                "units":  "Millimeters",
            },
            "imageDimensions": {"width": w, "height": h},
            "cameraParameterTables": [
                {
                    "parameterName": "FocalLengthTable",
                    "header":        "FocusEncoder, ZoomEncoder, Fx, Fy",
                    "data":          _ulens_data(fl_rows),
                },
                {
                    "parameterName": "ImageCenterTable",
                    "header":        "FocusEncoder, ZoomEncoder, Cx, Cy",
                    "data":          _ulens_data(ic_rows),
                },
                {
                    "parameterName": "NodalOffsetTable",
                    "header":        "FocusEncoder, ZoomEncoder, Qx, Qy, Qz, Qw, Tx, Ty, Tz",
                    "data":          _ulens_data(nd_rows),
                },
                {
                    "parameterName": "DistortionTable",
                    "header":        "FocusEncoder, ZoomEncoder, K1, K2, K3, P1, P2",
                    "data":          _ulens_data(dist_rows),
                },
            ],
            "encoderTables": [
                {"parameterName": "Focus", "header": "FocusEncoder, FocusCM",  "data": ""},
                {"parameterName": "Iris",  "header": "IrisEncoder, IrisFstop", "data": ""},
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(ulens, f, indent="\t")

        return {"success": True, "output_path": os.path.abspath(path), "error": None}
    except Exception as exc:
        return {"success": False, "output_path": path, "error": str(exc)}
