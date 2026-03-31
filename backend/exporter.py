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
) -> dict:
    try:
        cm = np.array(camera_matrix, dtype=np.float64)
        dc = np.array(dist_coeffs, dtype=np.float64).flatten()

        payload = {
            "calibration_date": datetime.now(timezone.utc).isoformat(),
            "rms": rms,
            "fov_x": fov_x,
            "fov_y": fov_y,
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

        map_x, map_y = cv2.initUndistortRectifyMap(
            cm, dc, None, cm, (w, h), cv2.CV_32FC1
        )

        # Normalize to 0–1 UV space (STmap convention)
        u = (map_x / (w - 1)).astype(np.float32)
        v = (map_y / (h - 1)).astype(np.float32)
        zeros = np.zeros((h, w), dtype=np.float32)

        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "dataWindow": ((0, 0), (w - 1, h - 1)),
            "displayWindow": ((0, 0), (w - 1, h - 1)),
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

        fx = float(cm[0, 0]) / w
        fy = float(cm[1, 1]) / h
        cx = float(cm[0, 2]) / w
        cy = float(cm[1, 2]) / h

        # OpenCV dist_coeffs order: k1 k2 p1 p2 [k3 ...]
        def _dc(i: int) -> float:
            return float(dc[i]) if len(dc) > i else 0.0

        k1, k2, p1, p2, k3 = _dc(0), _dc(1), _dc(2), _dc(3), _dc(4)

        # Static lens: single entry at focus=0.0, zoom=0.0
        focus_enc = 0.0
        zoom_enc  = 0.0

        ulens = {
            "metadata": {
                "type": "LensFile",
                "version": "0.0.0",
                "lensInfo": {
                    "serialNumber": "None",
                    "modelName": lens_name,
                    "distortionModel": "Spherical",
                },
                "name": lens_name,
                "nodalOffsetCoordinateSystem": "OpenCV",
                "fxFyUnits": "Normalized",
                "cxCyUnits": "Normalized",
                "userMetadata": [],
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
