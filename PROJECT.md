# Lens Calibration — Project Tracker

## Goals

- [ ] Deliver production-ready lens calibration tool for virtual production (UE5, Disguise, Pixotope) by 2026-07-01
- [ ] Support full zoom sweep calibration with PCHIP interpolation for broadcast zoom lenses by 2026-08-01
- [ ] Achieve reliable anamorphic/squeeze lens calibration workflow by 2026-09-01

## In Progress

- [ ] Polish live capture UI with pose-guided auto-capture for Blackmagic DeckLink and AJA devices
- [ ] Validate UE5 .ulens export format against real Unreal Engine virtual camera pipeline

## To Do

- [ ] Add support for additional capture devices (Magewell, Bluefish444 enumeration improvements)
- [ ] Implement batch file-based calibration workflow with per-FL focal-length grouping
- [ ] Build zoom sweep UI with dense interpolated calibration table visualization
- [ ] Add STmap EXR export validation with Nuke/After Effects
- [ ] Create end-user installer and Electron packaging for Windows distribution

## Done

- [x] FastAPI backend with WebSocket dispatcher for real-time calibration feedback
- [x] OpenCV checkerboard and ArUco (DICT_4x4_50, Sony AcuTarget) detection with quality scoring

## Blocked

- [ ] Some SDI capture card drivers require vendor SDK integration not yet licensed

## Releases

- v0.1.0 — planned 2026-06-01 — Initial release with single-FL calibration and OpenCV XML/JSON export

## Notes

- Architecture: Python FastAPI backend + Electron frontend (React/TypeScript)
- Physics-based focal-length initialization from fl_mm x sensor_width_px ensures reliable convergence at long focal lengths
