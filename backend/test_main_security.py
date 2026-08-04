"""
Tests for main.py — the WebSocket / export attack surface.

Browsers do not apply CORS to WebSockets, so the CORS middleware does nothing
for /ws: the Origin check in main.py is the only thing stopping a page the user
happens to visit from driving the backend on localhost. Exports are the only
write path reachable over that socket.

Covers:
  - Origin allowlist, including every origin the real app actually presents
    (packaged Electron file://, Vite dev server, LAN access from a phone)
  - Live /ws handshake: allowed origins can talk, others are refused
  - Export output-path validation (extension must match format, no traversal,
    parent directory must exist)
  - The export handler refuses a bad path without writing anything
  - A legitimate export still succeeds — the validator must not be so strict
    that it breaks the app
"""
import json
import os

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import main

try:  # newer starlette denies the handshake with an HTTP response
    from starlette.testclient import WebSocketDenialResponse
    _DENIED = (WebSocketDisconnect, WebSocketDenialResponse)
except ImportError:  # pragma: no cover — older starlette
    _DENIED = (WebSocketDisconnect,)


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


# ── Origin allowlist ─────────────────────────────────────────────────────────

class TestOriginAllowlist:
    """Origins the shipping app presents must be accepted, everything else not."""

    @pytest.mark.parametrize("origin", [
        None,                        # native clients (CLI tools, tests) send no Origin
        "file://",                   # packaged Electron — createWindow() uses loadFile()
        "null",                      # same, serialised as an opaque origin
        "http://localhost:5173",     # vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.42:5173",  # electron devUrl (LAN IP) + phone remote access
        "http://10.0.0.5:5173",
        "http://172.16.4.9:5173",
        "http://172.31.255.254",     # upper edge of the private 172.16/12 block
        "http://100.103.214.14:5173", # shared-address-space / CGNAT remote access
    ])
    def test_allowed(self, origin):
        assert main._is_allowed_origin(origin) is True

    @pytest.mark.parametrize("origin", [
        "http://evil.example.com",
        "https://evil.example.com",
        "http://localhost.evil.com",        # prefix look-alike
        "http://192.168.1.42.evil.com",     # private-IP look-alike
        "http://172.32.0.1:5173",           # just outside private 172.16/12
        "http://172.15.0.1:5173",           # just below it
        "https://localhost:5173",           # allowlist is http-only
    ])
    def test_rejected(self, origin):
        assert main._is_allowed_origin(origin) is False


class TestWebSocketHandshake:
    def test_allowed_origin_can_exchange_messages(self, client):
        with client.websocket_connect("/ws", headers={"origin": "file://"}) as ws:
            ws.send_text(json.dumps({"action": "ping"}))
            assert json.loads(ws.receive_text())["action"] == "ping"

    def test_rejected_origin_cannot_connect(self, client):
        # The handshake itself must fail. Do not read from the socket here: if
        # the check ever regresses the connection is accepted and a read would
        # block forever, turning a failure into a hung test run.
        with pytest.raises(_DENIED):
            with client.websocket_connect(
                "/ws", headers={"origin": "https://evil.example.com"}
            ):
                pass


# ── Export output-path validation ────────────────────────────────────────────

class TestExportPathValidation:
    def test_accepts_matching_extension(self, tmp_path):
        for fmt, ext in main._EXPORT_EXTENSIONS.items():
            path = str(tmp_path / f"out{ext}")
            assert main._check_output_path(path, fmt) is None, fmt

    def test_default_filenames_pass_their_own_validator(self):
        """The no-Electron fallback name must not be rejected by the validator."""
        for fmt, ext in main._EXPORT_EXTENSIONS.items():
            assert main._check_output_path(f"calibration{ext}", fmt) is None, fmt

    def test_relative_path_allowed(self):
        """Relative paths are the fallback when there is no native save dialog."""
        assert main._check_output_path("calibration.json", "json") is None

    def test_extension_must_match_format(self, tmp_path):
        assert main._check_output_path(str(tmp_path / "a.ulens"), "json") is not None
        assert main._check_output_path(str(tmp_path / "a.json"), "ue5_ulens") is not None

    @pytest.mark.parametrize("name", ["evil.bat", "evil.exe", "evil.ps1", "evil"])
    def test_executable_extensions_rejected(self, tmp_path, name):
        """The extension check is what stops a script landing in a startup folder."""
        assert main._check_output_path(str(tmp_path / name), "json") is not None

    def test_traversal_rejected(self):
        assert main._check_output_path("../../../evil.json", "json") is not None

    def test_missing_parent_directory_rejected(self, tmp_path):
        path = str(tmp_path / "no_such_dir" / "a.json")
        assert main._check_output_path(path, "json") is not None

    def test_empty_path_rejected(self):
        assert main._check_output_path("", "json") is not None


class TestExportHandler:
    def test_bad_extension_is_refused_and_nothing_written(self, client, tmp_path):
        # Payload is otherwise valid, so without the extension check this export
        # would succeed and write an executable — that is what is being blocked.
        target = tmp_path / "pwn.bat"
        with client.websocket_connect("/ws", headers={"origin": "file://"}) as ws:
            ws.send_text(json.dumps({
                "action": "export", "format": "json", "output_path": str(target),
                "camera_matrix": [[6400.0, 0, 1920.0], [0, 6400.0, 1080.0], [0, 0, 1]],
                "dist_coeffs": [-0.03, -0.01, 0.0, 0.0, 0.002],
                "fov_x": 33.4, "fov_y": 19.0, "rms": 0.41,
                "image_size": [3840, 2160],
            }))
            reply = json.loads(ws.receive_text())
        assert reply["action"] == "export_result"
        assert reply["success"] is False
        assert not target.exists()

    def test_unknown_format_refused(self, client, tmp_path):
        with client.websocket_connect("/ws", headers={"origin": "file://"}) as ws:
            ws.send_text(json.dumps({
                "action": "export", "format": "bogus",
                "output_path": str(tmp_path / "a.json"),
            }))
            reply = json.loads(ws.receive_text())
        assert reply["success"] is False

    def test_legitimate_export_still_succeeds(self, client, tmp_path):
        """Guards against the validator being tightened into breaking the app."""
        target = tmp_path / "calibration.json"
        with client.websocket_connect("/ws", headers={"origin": "file://"}) as ws:
            ws.send_text(json.dumps({
                "action": "export", "format": "json", "output_path": str(target),
                "camera_matrix": [[6400.0, 0, 1920.0], [0, 6400.0, 1080.0], [0, 0, 1]],
                "dist_coeffs": [-0.03, -0.01, 0.0, 0.0, 0.002],
                "fov_x": 33.4, "fov_y": 19.0, "rms": 0.41,
                "image_size": [3840, 2160],
            }))
            reply = json.loads(ws.receive_text())
        assert reply["success"] is True, reply.get("error")
        assert target.exists()


# ── Thumbnail read path ──────────────────────────────────────────────────────

class TestSafeReadPath:
    def test_traversal_rejected(self):
        assert main._is_safe_path("../../../etc/passwd") is False

    def test_nonexistent_file_rejected(self, tmp_path):
        assert main._is_safe_path(str(tmp_path / "nope.jpg")) is False

    def test_directory_rejected(self, tmp_path):
        assert main._is_safe_path(str(tmp_path)) is False

    def test_real_file_allowed(self, tmp_path):
        f = tmp_path / "frame.jpg"
        f.write_bytes(b"not-really-a-jpeg")
        assert main._is_safe_path(str(f)) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
