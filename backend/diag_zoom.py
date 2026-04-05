import glob
from zoom_calibrator import run_zoom_calibration
from frame_scorer import score_frame

board = (9, 6)
frames_28  = glob.glob('../captures/zoom_28mm/*.jpg')
frames_100 = glob.glob('../captures/zoom_100mm/*.jpg')
print(f'28mm files: {len(frames_28)}, 100mm files: {len(frames_100)}')

scored_28  = [score_frame(p, board) for p in frames_28]
scored_100 = [score_frame(p, board) for p in frames_100]

good_28  = [f | {'path': p} for f, p in zip(scored_28,  frames_28)  if f['quality'] != 'fail']
good_100 = [f | {'path': p} for f, p in zip(scored_100, frames_100) if f['quality'] != 'fail']
print(f'28mm usable: {len(good_28)}, 100mm usable: {len(good_100)}')
print(f'28mm det types:  {set(f["detection_type"] for f in good_28)}')
print(f'100mm det types: {set(f["detection_type"] for f in good_100)}')

result = run_zoom_calibration(
    [{'focal_length_mm': 28,  'frames': good_28},
     {'focal_length_mm': 100, 'frames': good_100}],
    board_cols=9, board_rows=6, square_size_mm=50,
    image_size=(1920, 1080),
    sensor_width_mm=36.0, sensor_height_mm=24.0,
)
for r in result['fl_results']:
    print(f"FL={r['focal_length_mm']}mm  rms={r.get('rms')}  error={r.get('error')}")
print(f"nodal_offsets: {result.get('nodal_offsets_mm')}")
