"""
Debug: show detected ArUco marker IDs, their image-space centres,
and compare them against our lookup table positions.
Sorted by physical board position so we can verify row/column layout.
"""
import sys, cv2, numpy as np
sys.path.insert(0, '.')
from frame_scorer import _SONY_BOARD_POSITIONS, _SONY_ID_START, _SONY_DICT_NAME

IMG = r"F:\Codebase\lenscalibration\captures\zoom_100mm\frame_1775260289527849400.jpg"

img = cv2.imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)
h_img, w_img = gray.shape

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
if hasattr(cv2.aruco, "ArucoDetector"):
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray_eq)
else:
    corners, ids, _ = cv2.aruco.detectMarkers(gray_eq, dictionary)

if ids is None:
    print("NO MARKERS DETECTED")
    sys.exit(1)

print(f"Image size: {w_img}×{h_img}")
print(f"Detected {len(ids)} markers\n")
print(f"{'ArucoID':>8}  {'rel':>4}  {'img_cx':>7}  {'img_cy':>7}  {'obj_x mm':>9}  {'obj_y mm':>9}  {'col':>5}  {'status'}")
print("-" * 75)

rows = []
for i, mid in enumerate(ids.flatten().tolist()):
    mid = int(mid)
    rel = mid - _SONY_ID_START
    img4 = np.array(corners[i], dtype=np.float32).reshape(4, 2)
    cx, cy = img4.mean(axis=0)
    if rel < 0 or rel >= len(_SONY_BOARD_POSITIONS):
        rows.append((mid, rel, cx, cy, None, None, "UNKNOWN ID"))
    else:
        ox, oy = _SONY_BOARD_POSITIONS[rel]
        rows.append((mid, rel, cx, cy, ox, oy, "ok"))

# Sort by obj_y then obj_x so we print in board order
rows.sort(key=lambda r: (r[5] if r[5] is not None else 9999,
                          r[4] if r[4] is not None else 9999))

prev_y = None
for mid, rel, cx, cy, ox, oy, status in rows:
    if prev_y is not None and oy is not None and oy != prev_y:
        print()
    prev_y = oy
    ox_s = f"{ox:9.1f}" if ox is not None else "       ?"
    oy_s = f"{oy:9.1f}" if oy is not None else "       ?"
    col_est = f"{cx/w_img*100:5.1f}%" if cx else "     -"
    print(f"{mid:>8}  {rel:>4}  {cx:7.1f}  {cy:7.1f}  {ox_s}  {oy_s}  {col_est}  {status}")

print(f"\nLookup table has {len(_SONY_BOARD_POSITIONS)} entries (IDs {_SONY_ID_START}–{_SONY_ID_START+len(_SONY_BOARD_POSITIONS)-1})")
