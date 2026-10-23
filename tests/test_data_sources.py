"""Tests for bounded public-data import and provenance metadata."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import socket

import pytest

from tis.data_sources import fetch_public_table_bytes


class FakeOpener:
    def __init__(self, response):
        self.response = response
    def open(self, request, timeout=30):
        return self.response


def fake_opener(monkeypatch, response):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: FakeOpener(response))


class FakeResponse:
    def __init__(self, payload: bytes, content_type: str = "text/csv", final_url: str = "https://example.org/data.csv"):
        self.payload = payload
        self.headers = {"Content-Type": content_type}
        self.final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.payload if size < 0 else self.payload[:size]

    def geturl(self):
        return self.final_url


def test_public_table_fetch_returns_hash_and_provenance(monkeypatch):
    payload = b"time_s,signal\n0,1\n1,2\n"
    fake_opener(monkeypatch, FakeResponse(payload))
    raw, manifest = fetch_public_table_bytes("https://example.org/data.csv")
    assert raw == payload
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["downloaded_bytes"] == str(len(payload))
    assert manifest["final_url"] == "https://example.org/data.csv"
    assert manifest["downloaded_at_utc"].endswith("+00:00")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/data.csv", "https://user:pass@example.org/data.csv", "/tmp/data.csv"])
def test_public_table_fetch_rejects_non_public_url_shapes(url):
    with pytest.raises(ValueError):
        fetch_public_table_bytes(url)


def test_public_table_fetch_rejects_html_response(monkeypatch):
    fake_opener(monkeypatch, FakeResponse(b"<!doctype html><html><body>login</body></html>", "text/html"))
    with pytest.raises(ValueError, match="HTML"):
        fetch_public_table_bytes("https://example.org/not-a-data-file")


def test_public_table_fetch_enforces_download_limit(monkeypatch):
    fake_opener(monkeypatch, FakeResponse(b"x" * 1025))
    with pytest.raises(ValueError, match="exceeds"):
        fetch_public_table_bytes("https://example.org/large.csv", max_bytes=1024)


def test_zarr_open_retries_transient_failure_and_passes_timeout():
    from tis.data_sources import _open_zarr_with_retry

    class FakeXarray:
        def __init__(self):
            self.calls = 0
        def open_zarr(self, url, group=None, storage_options=None):
            self.calls += 1
            assert storage_options == {"timeout": 12.0}
            if self.calls < 3:
                raise OSError("temporary archive error")
            return {"url": url, "group": group}

    fake = FakeXarray()
    result = _open_zarr_with_retry(fake, "https://example.org/shot.zarr", "magnetics", attempts=3, timeout_s=12)
    assert fake.calls == 3
    assert result["group"] == "magnetics"


def test_public_table_provenance_redacts_query_strings(monkeypatch):
    fake_opener(monkeypatch, FakeResponse(b"a,b\\n1,2\\n", final_url="https://example.org/data.csv?token=secret"))
    _, manifest = fetch_public_table_bytes("https://example.org/data.csv?api_key=secret")
    assert manifest["source_url"] == "https://example.org/data.csv"
    assert manifest["final_url"] == "https://example.org/data.csv"
    assert manifest["query_parameters_redacted"] is True
    assert "secret" not in str(manifest)
