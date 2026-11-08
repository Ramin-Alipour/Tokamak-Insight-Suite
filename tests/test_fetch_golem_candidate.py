from http.client import IncompleteRead
from urllib.error import HTTPError
from io import BytesIO

import pytest

from tools import fetch_golem_candidate as candidate


class _Response:
    def __init__(self, body=b"complete", url="https://example.test/data"):
        self._body = body
        self.headers = {"Content-Type": "text/plain"}
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def geturl(self):
        return self._url


def test_get_retries_incomplete_http_body(monkeypatch):
    calls = []
    responses = iter([
        _Response(IncompleteRead(b"partial", 20)),
        _Response(b"full validated response"),
    ])

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return next(responses)

    monkeypatch.setattr(candidate, "urlopen", fake_urlopen)
    monkeypatch.setattr(candidate, "sleep", lambda _: None)

    data, final_url, content_type = candidate.get(
        "https://example.test/data", attempts=3
    )

    assert data == b"full validated response"
    assert final_url == "https://example.test/data"
    assert content_type == "text/plain"
    assert len(calls) == 2


def test_get_does_not_retry_permanent_http_404(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(b""))

    monkeypatch.setattr(candidate, "urlopen", fake_urlopen)
    monkeypatch.setattr(candidate, "sleep", lambda _: None)

    with pytest.raises(HTTPError) as error:
        candidate.get("https://example.test/missing", attempts=4)

    assert error.value.code == 404
    assert len(calls) == 1


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_get_rejects_invalid_retry_count(attempts):
    with pytest.raises(ValueError, match="attempts"):
        candidate.get("https://example.test/data", attempts=attempts)
