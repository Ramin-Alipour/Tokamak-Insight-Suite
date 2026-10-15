#!/usr/bin/env python3
"""Reproduce selected EAST shot 22010 particle-flux statistics from public data.

This script reports raw-series statistics separately from detrended statistics.
It does not declare validation successful unless data units, magnetic field,
sign convention, and preprocessing can be matched to the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy import signal

RECORD = "https://zenodo.org/records/18473625"
FILES = {"ne": "22010ne.txt", "vf2": "22010vf2.txt", "vf3": "22010vf3.txt"}
README_NAME = "readme - EAST tokamak plasma probe data.docx"
README_URL = f"{RECORD}/files/readme%20-%20EAST%20tokamak%20plasma%20probe%20data.docx?download=1"
PAPER_DOI = "10.1088/1361-6587/ae4f20"
PUBLISHED_MEAN_FLUX = 1.3e20  # m^-2 s^-1; article figure 11 caption
PUBLISHED_SIGMA_FLUX = 2.6e20  # m^-2 s^-1; article figure 11 caption


def fetch(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Tokamak-Insight-Suite/empirical-validation"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} downloading {url}")
        return response.read()


def parse_numeric_series(raw: bytes, name: str) -> np.ndarray:
    try:
        values = np.loadtxt(raw.splitlines(), dtype=float, comments="#", ndmin=1)
    except Exception as exc:
        raise ValueError(f"{name}: expected a plain numeric text series: {exc}") from exc
    values = np.asarray(values, dtype=float).squeeze()
    if values.ndim != 1 or values.size < 100:
        raise ValueError(f"{name}: expected >=100 one-dimensional samples, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: non-finite values present; refusing silent deletion")
    return values


def extract_docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for node in root.findall(".//w:p", ns):
        paragraph = "".join(node.itertext()).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n".join(paragraphs)


def conditional_average(x: np.ndarray, y: np.ndarray, threshold_sigma: float = 1.5,
                        half_window: int = 200) -> dict:
    """Event-triggered conditional average of y around positive/negative x excursions."""
    x0 = signal.detrend(np.asarray(x, dtype=float), type="linear")
    y0 = signal.detrend(np.asarray(y, dtype=float), type="linear")
    scale = float(np.std(x0, ddof=1))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Conditioning signal has zero or invalid standard deviation")
    events = {}
    for label, condition in (
        ("positive", x0 >= threshold_sigma * scale),
        ("negative", x0 <= -threshold_sigma * scale),
    ):
        candidates = np.flatnonzero(condition)
        keep, last = [], -10**18
        min_distance = max(1, half_window // 2)
        for idx in candidates:
            if idx - last >= min_distance and idx >= half_window and idx + half_window < len(x0):
                keep.append(int(idx))
                last = int(idx)
        indices = np.asarray(keep, dtype=int)
        if indices.size:
            windows = np.stack([y0[i-half_window:i+half_window+1] for i in indices])
            avg = np.mean(windows, axis=0)
            events[label] = {"event_count": int(indices.size), "conditional_average": avg.tolist()}
        else:
            events[label] = {"event_count": 0, "conditional_average": []}
    return {"threshold_sigma": threshold_sigma, "half_window_samples": half_window, "events": events}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/east-22010-validation")
    parser.add_argument("--b-tesla", type=float, default=1.8,
                        help="Magnetic field in tesla; override for the relevant radial position.")
    parser.add_argument("--separation-m", type=float, default=0.008,
                        help="vf2-vf3 probe separation in metres (paper-stated 8 mm).")
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1,
                        help="Sign convention multiplier for E=(vf3-vf2)/separation.")
    parser.add_argument("--max-download-mb", type=float, default=200)
    args = parser.parse_args()
    if not math.isfinite(args.b_tesla) or args.b_tesla <= 0:
        parser.error("--b-tesla must be finite and > 0")
    if not math.isfinite(args.separation_m) or args.separation_m <= 0:
        parser.error("--separation-m must be finite and > 0")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    series, provenance = {}, {}
    for key, filename in FILES.items():
        url = f"{RECORD}/files/{filename}?download=1"
        raw = fetch(url)
        if len(raw) > args.max_download_mb * 1024 * 1024:
            raise RuntimeError(f"{filename} exceeds --max-download-mb safety limit")
        (outdir / filename).write_bytes(raw)
        series[key] = parse_numeric_series(raw, filename)
        provenance[key] = {
            "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "samples": int(series[key].size), "min": float(np.min(series[key])),
            "max": float(np.max(series[key])), "mean": float(np.mean(series[key])),
            "std_sample": float(np.std(series[key], ddof=1)),
        }

    lengths = {len(x) for x in series.values()}
    if len(lengths) != 1:
        raise ValueError(f"Signal lengths differ; no implicit truncation: "
                         f"{ {k: len(v) for k, v in series.items()} }")
    ne, vf2, vf3 = (series[k] for k in ("ne", "vf2", "vf3"))
    electric_field = args.sign * (vf3 - vf2) / args.separation_m
    flux = ne * electric_field / args.b_tesla
    flux_detrended = signal.detrend(flux, type="linear")

    # The published mean is compared with the raw derived flux, not the
    # detrended series (whose mean is near zero by construction).
    mean_flux = float(np.mean(flux))
    sigma_flux = float(np.std(flux, ddof=1))
    sigma_flux_detrended = float(np.std(flux_detrended, ddof=1))
    readme_text = ""
    readme_error = None
    try:
        readme_raw = fetch(README_URL)
        readme_text = extract_docx_text(readme_raw)
        (outdir / README_NAME).write_bytes(readme_raw)
    except Exception as exc:
        readme_error = f"{type(exc).__name__}: {exc}"

    mean_difference = abs(mean_flux - PUBLISHED_MEAN_FLUX) / abs(PUBLISHED_MEAN_FLUX) * 100
    sigma_difference = abs(sigma_flux - PUBLISHED_SIGMA_FLUX) / abs(PUBLISHED_SIGMA_FLUX) * 100
    report = {
        "title": "EAST shot 22010: measured-series particle-flux statistics",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_record": RECORD,
        "paper_doi": PAPER_DOI,
        "readme_url": README_URL,
        "dataset_readme_text": readme_text,
        "dataset_readme_error": readme_error,
        "method": {
            "density_series": FILES["ne"],
            "potential_series": [FILES["vf2"], FILES["vf3"]],
            "electric_field_formula": "E23=sign*(vf3-vf2)/Delta",
            "particle_flux_formula": "Gamma=ne*E23/B",
            "probe_separation_m": args.separation_m,
            "magnetic_field_T_supplied": args.b_tesla,
            "sign_multiplier": args.sign,
            "statistics": "raw mean and sample standard deviation; detrended standard deviation also reported",
            "limitation": "Unit scaling and exact preprocessing must be confirmed against the dataset documentation and paper.",
        },
        "source_files": provenance,
        "derived": {
            "samples": int(len(flux)),
            "electric_field_V_per_m_mean": float(np.mean(electric_field)),
            "electric_field_V_per_m_std": float(np.std(electric_field, ddof=1)),
            "flux_mean_raw_m-2_s-1": mean_flux,
            "flux_sigma_raw_m-2_s-1": sigma_flux,
            "flux_sigma_detrended_m-2_s-1": sigma_flux_detrended,
            "published_flux_mean_m-2_s-1": PUBLISHED_MEAN_FLUX,
            "published_flux_sigma_m-2_s-1": PUBLISHED_SIGMA_FLUX,
            "mean_relative_difference_percent": mean_difference,
            "sigma_raw_relative_difference_percent": sigma_difference,
            "flux_finite_fraction": float(np.mean(np.isfinite(flux))),
        },
        "conditional_averages": conditional_average(flux_detrended, flux_detrended),
        "acceptance": {
            "numeric_comparison_computed": True,
            "empirical_validation_pass": False,
            "reason": "Pending confirmation of source units, correct B at the probe position, sign convention, and matching paper preprocessing.",
        },
    }
    (outdir / "east_22010_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (outdir / "east_22010_flux.csv").open("w", encoding="utf-8") as f:
        f.write("sample,ne,vf2,vf3,electric_field_V_m,flux_raw_m-2_s-1,flux_detrended_m-2_s-1\n")
        for i in range(len(flux)):
            f.write(f"{i},{ne[i]:.12g},{vf2[i]:.12g},{vf3[i]:.12g},"
                    f"{electric_field[i]:.12g},{flux[i]:.12g},{flux_detrended[i]:.12g}\n")
    print(json.dumps({
        "report": str(outdir / "east_22010_validation.json"),
        "samples": len(flux), "flux_mean_raw": mean_flux,
        "flux_sigma_raw": sigma_flux, "flux_sigma_detrended": sigma_flux_detrended,
        "published_mean": PUBLISHED_MEAN_FLUX, "published_sigma": PUBLISHED_SIGMA_FLUX,
        "mean_relative_difference_percent": mean_difference,
        "sigma_raw_relative_difference_percent": sigma_difference,
        "empirical_validation_pass": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
