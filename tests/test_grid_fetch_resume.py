"""``grid_fetch._download`` completes a transfer the server cuts short.

Zenodo intermittently closes the connection before ``Content-Length`` bytes are sent, and a
chunked read then ends as if the file were complete (seen 2026-09-23: 62,427,980 of
66,823,560 bytes).  A local HTTP server reproduces that: its first response sends half the
file and closes; later requests honour ``Range``.  No network needed.
"""
from __future__ import annotations

import hashlib
import http.server
import threading

import pytest

from ceridwen.ssps.grid_fetch import _download

DATA = bytes(range(256)) * 4096            # 1 MiB


def _server(cut_first: int, honour_range: bool = True):
    state = {"n": 0, "ranges": []}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["n"] += 1
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if (rng and honour_range) else 0
            state["ranges"].append(rng)
            body = DATA[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(DATA)-1}/{len(DATA)}")
            self.end_headers()
            if state["n"] <= cut_first:
                self.wfile.write(body[: len(body) // 2])      # then drop the connection
                self.close_connection = True
                return
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


@pytest.mark.parametrize("cut_first", [0, 1, 3])
def test_short_transfer_is_resumed(tmp_path, cut_first):
    srv, state = _server(cut_first)
    try:
        out = tmp_path / "f.bin"
        _download(f"http://127.0.0.1:{srv.server_address[1]}/f", out)
        assert hashlib.sha256(out.read_bytes()).digest() == hashlib.sha256(DATA).digest()
        assert state["n"] == cut_first + 1
        assert all(r and r.startswith("bytes=") for r in state["ranges"][1:])
    finally:
        srv.shutdown()


def test_server_ignoring_range_restarts(tmp_path):
    srv, state = _server(cut_first=1, honour_range=False)
    try:
        out = tmp_path / "f.bin"
        _download(f"http://127.0.0.1:{srv.server_address[1]}/f", out)
        assert out.read_bytes() == DATA
    finally:
        srv.shutdown()


def test_gives_up_with_a_clear_error(tmp_path):
    srv, _state = _server(cut_first=100)
    try:
        with pytest.raises(RuntimeError, match="stopped at .* bytes after 3 attempts"):
            _download(f"http://127.0.0.1:{srv.server_address[1]}/f", tmp_path / "f.bin",
                      attempts=3)
    finally:
        srv.shutdown()


def test_404_fails_at_once(tmp_path):
    import urllib.error
    try:
        _download("https://zenodo.org/records/22921057/files/does-not-exist.h5?download=1",
                  tmp_path / "x", attempts=5)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    except OSError as exc:                                 # offline
        pytest.skip(f"no network: {exc}")
    else:
        pytest.fail("a missing file downloaded")
