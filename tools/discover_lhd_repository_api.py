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
    # Use the exact official form endpoint discovered in sumregdata2.js.
    # Retrieve both the measured CXS7 spectrum and NIFS-processed Ti products.
    target_diags = {"lhdcxs7_img", "lhdcxs7_prep", "lhdcxs7_cvi", "cxsmap7", "cxsmap7_ti"}
    api_base = "https://exp.lhd.nifs.ac.jp/opendata/LHD/dataexplog/"
    report["shot"] = 164208
    report["requested_diagnostics"] = sorted(target_diags)
    report["downloaded_files"] = []
    try:
        body = urllib.parse.urlencode({"SEARCH_SHOT_NO": "164208"}).encode("ascii")
        req = urllib.request.Request(
            urllib.parse.urljoin(api_base, "get_summary2.fcgi"),
            data=body,
            headers={**{"User-Agent": "Tokamak-Insight-Suite/1.0"},
                     "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=40) as response:
            summary_raw = response.read()
        (OUT / "shot_164208_summary.json").write_bytes(summary_raw)
        summary = json.loads(summary_raw.decode("utf-8", errors="replace"))
        report["shot_summary"] = {
            "status": summary.get("status"),
            "diagnostic_count": len(summary.get("arry_diag") or []),
            "response_file": "shot_164208_summary.json",
        }
        diag_names = summary.get("arry_diag") or []
        subnos = summary.get("arry_subno") or []
        report["available_target_diagnostics"] = []
        for i, diag in enumerate(diag_names):
            if diag not in target_diags or i >= len(subnos):
                continue
            subno = str(subnos[i])
            report["available_target_diagnostics"].append({"diagnostic": diag, "subno": subno})
            file_url = "https://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?" + urllib.parse.urlencode({
                "cmd": "getfile", "diag": diag, "shotno": "164208", "subno": subno
            })
            try:
                with urllib.request.urlopen(urllib.request.Request(
                    file_url, headers={"User-Agent": "Tokamak-Insight-Suite/1.0"}
                ), timeout=90) as response:
                    data = response.read()
                    content_type = response.headers.get("Content-Type", "")
                    disposition = response.headers.get("Content-Disposition", "")
                if len(data) > 250 * 1024 * 1024:
                    raise RuntimeError("Response exceeded 250 MiB limit")
                suffix = ".zip" if data[:4] == b"PK\\x03\\x04" else ".dat"
                filename = f"{diag}_164208_sub{subno}{suffix}"
                (OUT / filename).write_bytes(data)
                report["downloaded_files"].append({
                    "diagnostic": diag, "subno": subno, "url": file_url,
                    "status": 200, "content_type": content_type,
                    "content_disposition": disposition, "bytes": len(data),
                    "signature_hex": data[:16].hex(), "file": filename,
                })
            except Exception as e:
                report["downloaded_files"].append({
                    "diagnostic": diag, "subno": subno, "url": file_url,
                    "error": f"{type(e).__name__}: {e}",
                })
    except Exception as e:
        report["shot_summary_error"] = f"{type(e).__name__}: {e}"

    (OUT / "endpoint_strings.txt").write_text("\n".join(snippets), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(report["pages"]), "scripts": len(report["scripts"]),
                      "artifact_dir": str(OUT)}, indent=2))

if __name__ == "__main__":
    main()
