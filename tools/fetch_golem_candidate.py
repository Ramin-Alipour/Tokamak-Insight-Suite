#!/usr/bin/env python3
"""Fetch GOLEM shot-page spectrum/reference candidates with provenance.

This is an acquisition/inventory utility, not a scientific validation by itself.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from datetime import datetime, timezone

BASE = "https://golem.fjfi.cvut.cz/shots/{shot}/Data.php"
USER_AGENT = "Tokamak-Insight-Suite empirical-validation inventory/1.0"


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.href = None
        self.text_parts: list[str] = []
        self.items: list[dict[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.text_parts = []

    def handle_data(self, data):
        if self.href is not None:
            self.text_parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "a" and self.href is not None:
            self.items.append({"href": self.href, "text": " ".join(x for x in self.text_parts if x)})
            self.href = None
            self.text_parts = []


def get(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return response.read(), response.geturl()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=30622)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/golem-empirical-candidate"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    page_url = BASE.format(shot=args.shot)
    html, final_page_url = get(page_url)
    (out / "shot_page.html").write_bytes(html)
    parser_html = Links()
    parser_html.feed(html.decode("utf-8", errors="replace"))
    links = []
    for item in parser_html.items:
        absolute = urljoin(final_page_url, item["href"])
        if urlparse(absolute).scheme in {"http", "https"}:
            links.append({"text": item["text"], "url": absolute})

    # Save the full link inventory, then download only explicit links whose
    # displayed text or URL identifies the spectrum or temperature reference.
    (out / "links.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
    selected = []
    for item in links:
        token = (item["text"] + " " + item["url"]).lower()
        if "spectra.txt_data_" in token or ("temperature" in token and any(ext in token for ext in (".txt", ".xls", ".csv"))):
            if item["url"] not in {x["url"] for x in selected}:
                selected.append(item)

    downloads = []
    for i, item in enumerate(selected, start=1):
        try:
            data, final_url = get(item["url"])
            suffix = Path(urlparse(final_url).path).suffix or ".dat"
            name = f"download_{i:02d}{suffix}"
            (out / name).write_bytes(data)
            downloads.append({"label": item["text"], "source_url": item["url"], "final_url": final_url,
                              "file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "status": "downloaded"})
        except Exception as exc:  # inventory must retain per-link failures
            downloads.append({"label": item["text"], "source_url": item["url"], "status": "failed",
                              "error": f"{type(exc).__name__}: {exc}"})

    manifest = {
        "shot": args.shot,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "page_url": page_url,
        "final_page_url": final_page_url,
        "page_bytes": len(html),
        "page_sha256": hashlib.sha256(html).hexdigest(),
        "link_count": len(links),
        "selected_downloads": downloads,
        "classification": "candidate acquisition/inventory only; no rawness claim and no empirical validation claimed",
        "reference_value_seen_on_public_page_eV": 20.8 if args.shot == 30622 else None,
        "next_required_checks": [
            "Inspect file headers and acquisition/configuration metadata to determine raw versus corrected/processed status.",
            "Trace the archived temperature result to the exact input frames and analysis algorithm.",
            "Reproduce the same observable independently, then report absolute and relative discrepancy and uncertainty.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not downloads or not any(x.get("status") == "downloaded" for x in downloads):
        raise SystemExit("No candidate files downloaded; inspect links.json and manifest.json. This is not a validation pass.")


if __name__ == "__main__":
    main()
