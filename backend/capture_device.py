"""
Capture device enumeration and open helpers.

Detects professional capture cards (Blackmagic DeckLink, AJA, Bluefish444,
Magewell) by name, using DirectShow device enumeration on Windows.

Enumeration order
-----------------
1. pygrabber  (wraps ICreateDevEnum / IEnumMoniker — most reliable on Windows)
2. comtypes   (same COM path, no extra dependency if comtypes is installed)
3. PowerShell WMI fallback (PnP Camera class)
4. OpenCV brute-force scan 0-15 with CAP_DSHOW

Capture
-------
Always opens via cv2.CAP_DSHOW first (required for SDI capture cards on
Windows), falls back to default backend.  Applies the requested resolution
and frame-rate; for SDI cards the driver ignores these and returns the
locked-on signal resolution, so we read back actual W×H after opening.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Brand detection table
# ---------------------------------------------------------------------------

_BRANDS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r'blackmagic|decklink|ultrastudio|bmd|intensity\s*pro', re.I),
     'blackmagic', 'Blackmagic Design'),
    (re.compile(r'\baja\b|kona\b|io\s*xt|io\s*4k|t-tap|corvid|ttap', re.I),
     'aja', 'AJA'),
    (re.compile(r'bluefish|epoch\s*(4k|hdmi|horizon|supernova|neutron)|bermuda', re.I),
     'bluefish444', 'Bluefish444'),
    (re.compile(r'magewell|pro\s*capture|xi\s*capture', re.I),
     'magewell', 'Magewell'),
    (re.compile(r'datapath|vision\s*rgb|sc[- ]?4', re.I),
     'datapath', 'Datapath'),
    (re.compile(r'deltacast|delta-3g|delta-hd', re.I),
     'deltacast', 'DELTACAST'),
]

BRAND_ICONS: dict[str, str] = {
    'blackmagic': 'BMD',
    'aja':        'AJA',
    'bluefish444':'BF',
    'magewell':   'MW',
    'datapath':   'DP',
    'deltacast':  'DC',
    'generic':    'CAM',
}


def detect_brand(name: str) -> dict:
    for pattern, brand_id, brand_name in _BRANDS:
        if pattern.search(name):
            return {
                'id': brand_id,
                'name': brand_name,
                'icon': BRAND_ICONS[brand_id],
                'is_capture_card': True,
            }
    return {'id': 'generic', 'name': 'Generic / Webcam', 'icon': 'CAM', 'is_capture_card': False}


# ---------------------------------------------------------------------------
# Device enumeration backends
# ---------------------------------------------------------------------------

def _enum_pygrabber() -> list[str] | None:
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return None


def _enum_comtypes() -> list[str] | None:
    """Enumerate DirectShow video capture devices via comtypes COM calls."""
    try:
        import comtypes.client  # type: ignore
        import comtypes  # type: ignore

        CLSID_SystemDeviceEnum  = '{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}'
        CLSID_VideoInputDeviceCategory = '{860BB310-5D01-11d0-BD3B-00A0C911CE86}'
        IID_ICreateDevEnum   = '{29840822-5B84-11D0-BD3B-00A0C911CE86}'
        IID_IEnumMoniker     = '{00000102-0000-0000-C000-000000000046}'
        IID_IPropertyBag     = '{55272A00-42CB-11CE-8135-00AA004BB851}'

        dev_enum = comtypes.client.CreateObject(
            CLSID_SystemDeviceEnum,
            interface=comtypes.IUnknown,
        )
        # Narrow interface to ICreateDevEnum
        from comtypes.gen import devenum as _de  # noqa: F401 — may not exist
        return None  # comtypes path is fragile without generated stubs; skip
    except Exception:
        return None


def _enum_powershell() -> list[str] | None:
    """Fallback: ask PowerShell for PnP devices in Camera / Image / Media class."""
    try:
        ps = (
            "Get-PnpDevice -PresentOnly | "
            "Where-Object {$_.Class -in @('Camera','Image','Media')} | "
            "Select-Object -ExpandProperty FriendlyName | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        data = json.loads(result.stdout.strip())
        if isinstance(data, str):
            return [data]
        return list(data)
    except Exception:
        return None


def _enum_opencv_scan(max_index: int = 12) -> list[tuple[int, str]]:
    """Brute-force: try cv2 indices 0..max_index, return (index, label) pairs."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append((idx, f"Camera {idx}"))
            cap.release()
        else:
            # Stop after 3 consecutive misses past the first hit
            if found and idx > found[-1][0] + 2:
                break
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enumerate_capture_devices() -> list[dict]:
    """
    Return a list of detected video capture devices, each dict:
    {
        index: int,           # cv2.VideoCapture index
        name: str,            # human-readable device name
        brand: {id, name, icon, is_capture_card},
    }
    """
    devices: list[dict] = []

    # Try name-aware backends first
    names: list[str] | None = _enum_pygrabber()
    if names is None:
        names = _enum_powershell()

    if names:
        for idx, name in enumerate(names):
            devices.append({
                'index': idx,
                'name': name,
                'brand': detect_brand(name),
            })
        return devices

    # Fallback: OpenCV scan — no names available
    for idx, label in _enum_opencv_scan():
        devices.append({
            'index': idx,
            'name': label,
            'brand': detect_brand(label),
        })
    return devices


def open_capture(
    device_index: int,
    width:  int = 1920,
    height: int = 1080,
    fps:    int = 30,
) -> Optional[cv2.VideoCapture]:
    """
    Open a capture device, preferring DirectShow on Windows (required for
    SDI capture cards).  Requests the given resolution/fps; for locked SDI
    signals the driver overrides these — read back actual size after opening.
    Returns None if the device cannot be opened.
    """
    # Prefer DirectShow backend (best compatibility with capture cards on Windows)
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        return None

    # Minimise buffer depth so frames are as fresh as possible
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Request format — capture cards may ignore these and lock to signal
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS,          float(fps))

    return cap


def read_actual_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Return (width, height) as reported by the opened capture device."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w or 1920, h or 1080


def read_actual_fps(cap: cv2.VideoCapture, n_frames: int = 6) -> float:
    """
    Return the actual signal fps.
    CAP_PROP_FPS on DirectShow often returns the driver default (30.0) even
    when the SDI signal is 24 or 25 fps.  We measure by timing n_frames live
    frames and only fall back to the reported value if measurement is unreliable.
    Keeping n_frames small (default 6) minimises startup latency.
    """
    reported = cap.get(cv2.CAP_PROP_FPS)

    t0 = time.perf_counter()
    count = 0
    for _ in range(n_frames):
        ret, _ = cap.read()
        if ret:
            count += 1
    elapsed = time.perf_counter() - t0

    if count >= 4 and elapsed > 0.05:
        measured = count / elapsed
        standards = [23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0]
        snapped = min(standards, key=lambda s: abs(s - measured))
        if abs(snapped - measured) < 2.0:
            return snapped
        return round(measured, 3)

    return reported if reported and reported > 0 else 30.0
