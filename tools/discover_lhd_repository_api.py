#!/usr/bin/env python3
"""Download the public NIFS LHD analyzed-data search page and its referenced JavaScript."""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

URLS = [
    "https://exp.lhd.nifs.ac.jp/opendata/LHD/dataexplog/",
    "https://exp.lhd.nifs.ac.jp/opendata/LHD/dataexplog/registered_data_data_name.html",
]
OUT = Path("artifacts/lhd-repository-discovery")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Tokamak-Insight-Suite/1.0"})
    with urllib.request.urlopen(req, timeout=25) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"pages": [], "scripts": []}
    scripts = set()
    for i, url in enumerate(URLS):
        try:
            status, content_type, raw = get(url)
            name = f"page_{i}.html"
            (OUT / name).write_bytes(raw)
            page = raw.decode("utf-8", errors="replace")
            report["pages"].append({"url": url, "status": status, "content_type": content_type,
                                    "bytes": len(raw), "file": name})
            for src in re.findall(r'<script[^>]+src=["\']([^"\']+)', page, re.I):
                scripts.add(urllib.parse.urljoin(url, src))
        except Exception as e:
            report["pages"].append({"url": url, "error": str(e)})
    snippets = []
    for i, url in enumerate(sorted(scripts)):
        try:
            status, content_type, raw = get(url)
            name = f"script_{i}.js"
            (OUT / name).write_bytes(raw)
            page = raw.decode("utf-8", errors="replace")
            report["scripts"].append({"url": url, "status": status, "content_type": content_type,
                                      "bytes": len(raw), "file": name})
            for line_no, line in enumerate(page.splitlines(), 1):
                if re.search(r"(?i)(ajax|api|search|download|\.php|\.cgi|shotno|shot_number|data_name)", line):
                    snippets.append(f"### {url}:{line_no}\n{line[:1000]}")
        except Exception as e:
            report["scripts"].append({"url": url, "error": str(e)})
    (OUT / "endpoint_strings.txt").write_text("\n".join(snippets), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(report["pages"]), "scripts": len(report["scripts"]),
                      "artifact_dir": str(OUT)}, indent=2))

if __name__ == "__main__":
    main()
