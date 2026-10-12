"""
Tests for the dataset downloader.

These exercise the caching, retry and resume behaviour with a stubbed
``requests.get``, so they need no network and run in the default suite.
"""

from __future__ import annotations

import pytest
import requests

from polyclustx.data import downloader
from polyclustx.data.downloader import download_file

PAYLOAD = b"0123456789" * 10  # 100 bytes


class FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, body: bytes, status_code: int = 200, total: int | None = None):
        self._body = body
        self.status_code = status_code
        self.headers = {"content-length": str(len(body) if total is None else total)}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retries should not actually wait during tests."""
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)


def test_returns_cached_file_without_requesting(tmp_path, monkeypatch):
    dest = tmp_path / "data.mat"
    dest.write_bytes(b"cached")

    def _boom(*a, **kw):
        raise AssertionError("network should not be touched for a cached file")

    monkeypatch.setattr(requests, "get", _boom)

    assert download_file("http://example/x.mat", dest) == dest
    assert dest.read_bytes() == b"cached"


def test_downloads_and_moves_into_place(tmp_path, monkeypatch):
    dest = tmp_path / "data.mat"
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse(PAYLOAD))

    download_file("http://example/x.mat", dest)

    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_suffix(".mat.part").exists()


def test_retries_after_a_dropped_connection(tmp_path, monkeypatch):
    dest = tmp_path / "data.mat"
    calls = []

    def _get(url, **kw):
        calls.append(kw.get("headers") or {})
        if len(calls) == 1:
            raise requests.exceptions.ChunkedEncodingError("connection broken")
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(requests, "get", _get)

    download_file("http://example/x.mat", dest, retries=3)

    assert len(calls) == 2
    assert dest.read_bytes() == PAYLOAD


def test_resumes_from_partial_data(tmp_path, monkeypatch):
    dest = tmp_path / "data.mat"
    calls = []

    def _get(url, **kw):
        headers = kw.get("headers") or {}
        calls.append(headers)
        if len(calls) == 1:
            # Deliver half the body, then fail the size check.
            return FakeResponse(PAYLOAD[:50], total=len(PAYLOAD))
        # Second attempt asks for the rest and the server honours it.
        assert headers.get("Range") == "bytes=50-"
        return FakeResponse(PAYLOAD[50:], status_code=206)

    monkeypatch.setattr(requests, "get", _get)

    download_file("http://example/x.mat", dest, retries=3)

    assert len(calls) == 2
    assert dest.read_bytes() == PAYLOAD


def test_restarts_when_server_ignores_the_range(tmp_path, monkeypatch):
    dest = tmp_path / "data.mat"
    calls = []

    def _get(url, **kw):
        calls.append(kw.get("headers") or {})
        if len(calls) == 1:
            return FakeResponse(PAYLOAD[:50], total=len(PAYLOAD))
        return FakeResponse(PAYLOAD)          # 200, not 206 — start over

    monkeypatch.setattr(requests, "get", _get)

    download_file("http://example/x.mat", dest, retries=3)

    assert dest.read_bytes() == PAYLOAD       # not appended to the partial data


def test_raises_and_cleans_up_when_retries_run_out(tmp_path, monkeypatch):
    dest = tmp_path / "data.mat"

    def _get(*a, **kw):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "get", _get)

    with pytest.raises(requests.exceptions.ReadTimeout):
        download_file("http://example/x.mat", dest, retries=2)

    assert not dest.exists()
    assert not dest.with_suffix(".mat.part").exists()
