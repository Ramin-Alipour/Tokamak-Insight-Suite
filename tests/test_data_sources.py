"""Tests for bounded public-data import and provenance metadata."""
from __future__ import annotations

import hashlib
import io
import urllib.request

import pytest

from tis.data_sources import fetch_public_table_bytes


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
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload))
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
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b"<!doctype html><html><body>login</body></html>", "text/html"),
    )
    with pytest.raises(ValueError, match="HTML"):
        fetch_public_table_bytes("https://example.org/not-a-data-file")


def test_public_table_fetch_enforces_download_limit(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b"x" * 1025),
    )
    with pytest.raises(ValueError, match="exceeds"):
        fetch_public_table_bytes("https://example.org/large.csv", max_bytes=1024)
