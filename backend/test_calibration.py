"""
Diagnostic script — runs frame scoring + calibration on test captures.

Usage (from backend/):
    python test_calibration.py

Tested folders:
    ../captures/zoom_28mm   -> 28 mm focal length
    ../captures/zoom_100mm  -> 100 mm focal length

Default board: 9 x 6 inner corners, 25 mm squares.
Override with --cols, --rows, --sq.
"""

import argparse
import os
import sys
import json
from pprint import pprint

import cv2

# ── make sure the backend package is importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from frame_scorer import score_frame
from calibrator import run_calibration


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_size(folder: str, paths: list) -> tuple:
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    return (1920, 1080)


def _score_folder(folder: str, board_size: tuple, verbose: bool) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not paths:
        print(f"  [!] No image files found in {folder}")
        return []

    print(f"\n  Scoring {len(paths)} image(s) with board {board_size[0]}x{board_size[1]} ...")
    scored = []
    for p in paths:
        result = score_frame(p, board_size)
        result["path"] = p
        flag = "OK  " if result.get("quality") != "fail" else "FAIL"
        det  = result.get("detection_type") or "—"
        print(f"    [{flag}] {os.path.basename(p):50s}  "
              f"quality={result.get('quality'):8s}  "
              f"det={det:12s}  "
              f"sharpness={result.get('sharpness', 0):8.1f}  "
              f"coverage={result.get('coverage', 0):.3f}  "
              f"reason={result.get('reason', '')}")
        if verbose:
            pprint(result)
        scored.append(result)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def run_test(folder: str, label: str, board_cols: int, board_rows: int,
             square_size_mm: float, verbose: bool) -> None:
    print(f"\n{'='*70}")
    print(f"  Focal length set: {label}")
    print(f"  Folder         : {folder}")
    print(f"  Board          : {board_cols}x{board_rows} inner corners, {square_size_mm} mm/sq")
    print(f"{'='*70}")

    if not os.path.isdir(folder):
        print(f"  [!] Folder not found: {folder}")
        return

    board_size = (board_cols, board_rows)

    # 1. score frames
    scored = _score_folder(folder, board_size, verbose)
    if not scored:
        print("  [!] No frames scored — cannot calibrate.")
        return

    total   = len(scored)
    usable  = [f for f in scored if f.get("quality") != "fail" and f.get("corners")]
    fails   = [f for f in scored if f.get("quality") == "fail"]
    det_types = set(f.get("detection_type") for f in usable if f.get("detection_type"))

    print(f"\n  Summary: {total} frames, {len(usable)} usable, {len(fails)} failed")
    print(f"  Detection types in usable set: {det_types or {'none'}}")

    if len(usable) < 3:
        print("\n  [!] Fewer than 3 usable frames — cannot calibrate.")
        print("  Possible causes:")
        print("    • Checkerboard pattern not detected (board size mismatch?)")
        print("    • Images too blurry or overexposed")
        print("    • Try different --cols / --rows values")
        return

    # 2. infer image size from first readable image in folder
    image_paths = [f["path"] for f in scored]
    image_size  = _image_size(folder, image_paths)
    print(f"  Image size: {image_size[0]} x {image_size[1]}")

    # 3. run calibration
    print(f"\n  Running calibration ...")
    result = run_calibration(
        scored_frames=usable,
        board_cols=board_cols,
        board_rows=board_rows,
        square_size_mm=square_size_mm,
        image_size=image_size,
    )

    # 4. report
    if result.get("error"):
        print(f"\n  [CALIBRATION FAILED]\n  Error: {result['error']}")
    else:
        print(f"\n  [CALIBRATION SUCCEEDED]")
        print(f"  RMS reprojection error : {result.get('rms'):.4f} px")
        print(f"  Confidence             : {result.get('confidence')}")
        print(f"  Used / skipped frames  : {result.get('used_frames')} / {result.get('skipped_frames')}")
        print(f"  FOV x / y              : {result.get('fov_x'):.2f}° / {result.get('fov_y'):.2f}°")
        cm = result.get("camera_matrix")
        if cm:
            print(f"  fx={cm[0][0]:.2f}  fy={cm[1][1]:.2f}  cx={cm[0][2]:.2f}  cy={cm[1][2]:.2f}")
        print(f"  Dist coeffs: {result.get('dist_coeffs')}")
        if result.get("solver_warning"):
            print(f"  Warning: {result['solver_warning']}")
        print(f"\n  Per-image reprojection errors:")
        for e in result.get("per_image_errors", []):
            flag = " <-- OUTLIER" if e.get("outlier") else ""
            print(f"    {os.path.basename(e['path']):50s}  {e['error']:.4f} px{flag}")


def main():
    parser = argparse.ArgumentParser(description="Lens calibration diagnostics")
    parser.add_argument("--cols",   type=int,   default=7,    help="Inner corners cols (default 7 for Sony 8×6 board)")
    parser.add_argument("--rows",   type=int,   default=5,    help="Inner corners rows (default 5 for Sony 8×6 board)")
    parser.add_argument("--sq",     type=float, default=50.0, help="Square size in mm (default 50 — Sony AcuTarget)")
    parser.add_argument("--verbose",action="store_true",      help="Print full frame result dicts")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..", "captures")

    run_test(
        folder=os.path.join(base, "zoom_28mm"),
        label="28 mm",
        board_cols=args.cols,
        board_rows=args.rows,
        square_size_mm=args.sq,
        verbose=args.verbose,
    )
    run_test(
        folder=os.path.join(base, "zoom_100mm"),
        label="100 mm",
        board_cols=args.cols,
        board_rows=args.rows,
        square_size_mm=args.sq,
        verbose=args.verbose,
    )

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
