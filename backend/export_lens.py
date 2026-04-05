"""
export_lens.py — Score frames, run zoom calibration, export .ulens with nodal offsets.

Usage:
    python export_lens.py [options]

Options:
    --lens-name NAME        Lens name embedded in the file (default: "Lens")
    --sensor-w MM           Sensor width in mm (default: 0.0 = normalised only)
    --sensor-h MM           Sensor height in mm (default: 0.0)
    --out PATH              Output .ulens path (default: ../lens_output.ulens)
    --json PATH             Also export a per-FL JSON summary (optional)
    --cols N                Board inner corners wide (default: 7)
    --rows N                Board inner corners tall (default: 5)
    --sq MM                 Square size in mm (default: 50.0)
    --nodal-preset NAME     Use a named preset from nodal_presets.json.
    --nodal-file PATH       Load a custom presets JSON file instead of the built-in one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Manufacturer nodal offset presets — loaded from nodal_presets.json
# Format: { "preset-name": { "description": "...", "points": [[fl_mm, nodal_mm], ...] } }
# Use --nodal-file to point to a different JSON file.
# ---------------------------------------------------------------------------
_PRESETS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodal_presets.json")

from frame_scorer import score_frame
from zoom_calibrator import run_zoom_calibration
from exporter import export_ue5_ulens_zoom, load_nodal_presets, apply_nodal_preset, apply_fl_override

# Re-export the shared helpers under the old names so the rest of this file works unchanged
NODAL_PRESETS: dict[str, dict] = load_nodal_presets(_PRESETS_DEFAULT)

def _apply_nodal_preset(preset_name, fl_results, fl_interpolated, nodal_offsets_mm):
    apply_nodal_preset(preset_name, fl_results, fl_interpolated, nodal_offsets_mm, presets=NODAL_PRESETS)

def _apply_fl_override(preset_name, fl_results, fl_interpolated, sensor_width_mm, image_width_px):
    apply_fl_override(preset_name, fl_results, fl_interpolated, sensor_width_mm, image_width_px, presets=NODAL_PRESETS)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures")
_IMAGE_SIZE = (1920, 1080)
_BOARD_COLS = 7
_BOARD_ROWS = 5
_SQ_MM = 50.0


def _discover_fl_dirs(base_dir: str) -> dict[int, str]:
    """Scan *base_dir* for subfolders named zoom_<N>mm and return {FL_mm: folder_name}."""
    result: dict[int, str] = {}
    if not os.path.isdir(base_dir):
        return result
    for name in os.listdir(base_dir):
        m = re.match(r'^zoom_(\d+)mm$', name)
        if m and os.path.isdir(os.path.join(base_dir, name)):
            result[int(m.group(1))] = name
    return dict(sorted(result.items()))


def _collect_frames(capture_dir: str, board_size: tuple) -> list[dict]:
    """Score all images in *capture_dir* and return the scored-frame dicts."""
    patterns = [
        os.path.join(capture_dir, "*.jpg"),
        os.path.join(capture_dir, "*.jpeg"),
        os.path.join(capture_dir, "*.png"),
        os.path.join(capture_dir, "*.tiff"),
        os.path.join(capture_dir, "*.tif"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths.sort()

    if not paths:
        print(f"  [warn] No images found in {capture_dir}")
        return []

    frames = []
    for p in paths:
        result = score_frame(p, board_size)
        quality = result.get("quality", "fail")
        det = result.get("detection_type", "-")
        n = len(result.get("corners") or [])
        print(f"    {os.path.basename(p):30s}  quality={quality:<6}  det={det:<15}  pts={n}")
        frames.append(result)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lens calibration to .ulens")
    parser.add_argument("--lens-name", default="Lens")
    parser.add_argument("--sensor-w", type=float, default=0.0)
    parser.add_argument("--sensor-h", type=float, default=0.0)
    parser.add_argument("--out", default=os.path.join(_BASE, "..", "lens_output.ulens"))
    parser.add_argument("--json", default=None, help="Optional JSON summary output path")
    parser.add_argument("--cols", type=int, default=_BOARD_COLS)
    parser.add_argument("--rows", type=int, default=_BOARD_ROWS)
    parser.add_argument("--sq", type=float, default=_SQ_MM)
    parser.add_argument(
        "--nodal-file", default=None, metavar="PATH",
        help=f"Path to a nodal-presets JSON file (default: nodal_presets.json next to this script)"
    )
    parser.add_argument(
        "--nodal-preset", default=None, metavar="NAME",
        help="Override calibrated nodal offsets with a named preset from the presets file."
    )
    args = parser.parse_args()

    # Reload presets now so --nodal-file is respected before validation
    global NODAL_PRESETS
    if args.nodal_file:
        NODAL_PRESETS = load_nodal_presets(os.path.abspath(args.nodal_file))

    if args.nodal_preset and args.nodal_preset not in NODAL_PRESETS:
        print(f"[error] Unknown nodal preset '{args.nodal_preset}'.")
        print(f"  Available: {', '.join(NODAL_PRESETS) or '(none)'}")
        sys.exit(1)

    board_size = (args.cols, args.rows)
    out_path = os.path.abspath(args.out)

    print("=" * 60)
    print("Lens Calibration Export")
    print(f"  Board: {args.cols}×{args.rows} inner corners, {args.sq}mm/sq")
    print(f"  Output: {out_path}")
    print("=" * 60)

    fl_groups = []
    discovered = _discover_fl_dirs(_BASE)
    if not discovered:
        print(f"\n[error] No zoom_<N>mm folders found in: {os.path.abspath(_BASE)}")
        print("  Create folders like captures/zoom_28mm/, captures/zoom_50mm/, captures/zoom_100mm/")
        sys.exit(1)
    for fl_mm, dir_name in discovered.items():
        capture_dir = os.path.join(_BASE, dir_name)
        print(f"\nScoring {fl_mm}mm frames from: {capture_dir}")
        frames = _collect_frames(capture_dir, board_size)
        fl_groups.append({"focal_length_mm": fl_mm, "frames": frames})

    # -----------------------------------------------------------------------
    print("\nRunning zoom calibration …")
    result = run_zoom_calibration(
        fl_groups=fl_groups,
        board_cols=args.cols,
        board_rows=args.rows,
        square_size_mm=args.sq,
        image_size=_IMAGE_SIZE,
        sensor_width_mm=args.sensor_w,
        sensor_height_mm=args.sensor_h,
        squeeze_ratio=1.0,
    )

    if not result["success"]:
        print(f"[error] Zoom calibration failed: {result.get('error')}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    print("\nCalibration results:")
    print(f"  {'FL':>6}  {'RMS':>7}  {'Confidence':<12}  {'Frames':>6}  {'fx':>8}  {'fy':>8}  Nodal-Z")
    nodal = result["nodal_offsets_mm"]
    for r in result["fl_results"]:
        fl  = r["focal_length_mm"]
        rms = r.get("rms")
        if rms is None:
            print(f"  {fl:>6.0f}mm  FAILED  {r.get('error', '')}")
            continue
        key = str(int(fl)) if fl == int(fl) else f"{fl:.1f}"
        nz  = nodal.get(key, 0.0)
        print(
            f"  {fl:>6.0f}mm  {rms:>6.2f}px  {r.get('confidence','?'):<12}"
            f"  {r.get('used_frames',0):>6}  {r.get('fx_px',0):>8.1f}  {r.get('fy_px',0):>8.1f}"
            f"  {nz:+.2f}mm"
        )
        if r.get("warning"):
            print(f"          [!] {r['warning']}")

    print(f"\nNodal offsets (mm, Z-shift vs best FL): {nodal}")
    nm = result.get("nodal_model")
    if nm:
        print(f"Nodal model type: {nm.get('model_type', '?')}")

    # -----------------------------------------------------------------------
    if args.nodal_preset:
        print(f"\nApplying nodal preset: {args.nodal_preset}")
        _apply_nodal_preset(
            args.nodal_preset,
            result["fl_results"],
            result.get("fl_interpolated"),
            nodal,
        )
        print(f"Updated nodal offsets (mm): {nodal}")

        if args.sensor_w > 0:
            _apply_fl_override(
                args.nodal_preset,
                result["fl_results"],
                result.get("fl_interpolated"),
                sensor_width_mm=args.sensor_w,
                image_width_px=_IMAGE_SIZE[0],
            )

    # -----------------------------------------------------------------------
    print(f"\nExporting .ulens -> {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    export_result = export_ue5_ulens_zoom(
        path=out_path,
        fl_results=result["fl_results"],
        image_size=_IMAGE_SIZE,
        nodal_offsets_mm=nodal,
        lens_name=args.lens_name,
        sensor_width_mm=args.sensor_w,
        sensor_height_mm=args.sensor_h,
        squeeze_ratio=1.0,
        lens_type="spherical",
        fl_interpolated=result.get("fl_interpolated"),
    )

    if export_result["success"]:
        print(f"[ok] Written: {export_result['output_path']}")
    else:
        print(f"[error] Export failed: {export_result['error']}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    if args.json:
        json_path = os.path.abspath(args.json)
        summary = {
            "fl_results":       result["fl_results"],
            "nodal_offsets_mm": result["nodal_offsets_mm"],
            "nodal_model":      result.get("nodal_model"),
        }
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[ok] JSON summary: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
