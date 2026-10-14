#!/usr/bin/env python3
"""Independent external benchmark: EAST probe time series vs published statistics.

Source: Zenodo DOI 10.5281/zenodo.18473625 and its associated 2026 paper.
This script does not use papers authored by Ramin Alipour.
It preserves source values, computes window statistics, and reports comparison
status only after unit/scale conversion has been explicitly established.
"""
from __future__ import annotations
import json, re, urllib.request, zipfile, io
from pathlib import Path
import numpy as np

BASE = "https://zenodo.org/records/18473625/files/"
FILES = ["22010ne.txt", "22010Te.txt", "22010vf1.txt", "22010vf2.txt", "22010vf3.txt"]
OUT = Path("artifacts/east-probe-validation")
OUT.mkdir(parents=True, exist_ok=True)

def fetch(name: str) -> bytes:
    req = urllib.request.Request(BASE + name + "?download=1", headers={"User-Agent": "Tokamak-Insight-Suite-validation/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()

def numbers(raw: bytes) -> np.ndarray:
    return np.loadtxt(io.BytesIO(raw), dtype=float, ndmin=1).reshape(-1)

# Published analysis window is data points 275000--309600 (paper: 275.000--309.600).
# Python slice is end-exclusive, matching the stated interval.
start, stop = 275000, 309600
expected = {
    "ne": {"std": 39e16, "unit": "m^-3", "paper": "sigma_ne = 39 x 10^16 m^-3"},
    "Te": {"std": 9.6, "unit": "eV", "paper": "sigma_Te = 9.6 eV"},
    "vf1": {"std": 8.7, "unit": "V", "paper": "sigma_vf1 = 8.7 V"},
    "vf2": {"std": 7.9, "unit": "V", "paper": "sigma_vf2 = 7.9 V"},
    "vf3": {"std": 6.8, "unit": "V", "paper": "sigma_vf3 = 6.8 V"},
}
results = []
source_units = {}
for name in FILES:
    raw = fetch(name)
    (OUT / name).write_bytes(raw)
    x = numbers(raw)
    if len(x) < stop:
        raise RuntimeError(f"{name}: {len(x)} rows, fewer than required {stop}")
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{name}: non-finite values in source series")
    key = name.removeprefix("22010").removesuffix(".txt")
    segment = x[start:stop]
    stats = {
        "file": name, "n_total": int(len(x)), "window_start_index": start,
        "window_stop_exclusive": stop, "n_window": int(len(segment)),
        "raw_mean": float(np.mean(segment)), "raw_std_sample": float(np.std(segment, ddof=1)),
        "raw_min": float(np.min(segment)), "raw_max": float(np.max(segment)),
        "raw_unit_scale_confirmed": False,
        "published_reference": expected[key]["paper"],
        "comparison_status": "NOT_COMPARABLE_YET: source-unit/scale mapping must be confirmed from README/methods before numerical agreement is scored",
    }
    results.append(stats)
    print(f"DATA PASS {name}: rows={len(x)}, window_n={len(segment)}, raw_std={stats['raw_std_sample']:.9g}; published={expected[key]['paper']}")

# Download and extract the dataset's own README so unit conventions can be checked.
readme_name = "readme - EAST tokamak plasma probe data.docx"
readme_bytes = fetch(readme_name)
(OUT / "dataset_readme.docx").write_bytes(readme_bytes)
try:
    with zipfile.ZipFile(io.BytesIO(readme_bytes)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    text = re.sub(r"</w:p>", "\\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&apos;", "'"))
    (OUT / "dataset_readme_extracted.txt").write_text(text, encoding="utf-8")
    print("README extracted for unit/scale review.")
except Exception as exc:
    print(f"README extraction warning: {exc}")

report = {
    "dataset": "EAST tokamak plasma probe data",
    "doi": "10.5281/zenodo.18473625",
    "paper_window_indices_python": [start, stop],
    "results": results,
    "status": "RAW_DATA_RETRIEVAL_AND_STATISTICS_PASS; PHYSICAL_COMPARISON_PENDING_UNIT_SCALE_CONFIRMATION",
    "scientific_rule": "No pass/fail physical agreement is assigned from mismatched or unconfirmed units.",
}
(OUT / "east_probe_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("SUMMARY: raw data retrieval and independent window statistics completed; unit-aware comparison is not falsely claimed.")
