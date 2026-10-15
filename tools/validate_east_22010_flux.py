#!/usr/bin/env python3
"""Reproduce selected EAST shot 22010 turbulence statistics from public Zenodo files.

The script separates an executable numerical comparison from a claim of full
reproduction: the paper's reported flux standard deviation is a reference value,
while magnetic field B and preprocessing choices must be supplied explicitly.
"""
from __future__ import annotations
import argparse, hashlib, json, math, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import signal

RECORD = "https://zenodo.org/records/18473625"
FILES = {"ne": "22010ne.txt", "vf2": "22010vf2.txt", "vf3": "22010vf3.txt"}
PAPER_DOI = "10.1088/1361-6587/ae4f20"
PUBLISHED_MEAN_FLUX = 1.3e20  # m^-2 s^-1; article figure 11 caption
PUBLISHED_SIGMA_FLUX = 2.6e20  # m^-2 s^-1; article figure 11 caption\nREADME_URL = f"{RECORD}/files/readme%20-%20EAST%20tokamak%20plasma%20probe%20data.docx?download=1"

def fetch(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Tokamak-Insight-Suite/empirical-validation"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} downloading {url}")
        return response.read()

def parse_numeric_series(raw: bytes, name: str) -> np.ndarray:
    try:
        values = np.loadtxt(raw.splitlines(), dtype=float, comments="#", ndmin=1)
    except Exception as exc:
        raise ValueError(f"{name}: expected a plain numeric text series; parsing failed: {exc}") from exc
    values = np.asarray(values, dtype=float).squeeze()
    if values.ndim != 1 or values.size < 100:
        raise ValueError(f"{name}: expected >=100 one-dimensional samples, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: non-finite values are present; refusing silent deletion")
    return values

def extract_docx_text(raw: bytes) -> str:
    import zipfile
    import xml.etree.ElementTree as ET
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        xml = archive.read('word/document.xml')
    root = ET.fromstring(xml)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = []
    for node in root.findall('.//w:p', ns):
        paragraph = ''.join(node.itertext()).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return chr(10).join(paragraphs)

def conditional_average(x: np.ndarray, y: np.ndarray, threshold_sigma: float = 1.5,
                        half_window: int = 200) -> dict:
    """Event-triggered conditional average of y around positive/negative x excursions."""
    x0 = signal.detrend(np.asarray(x, dtype=float), type="linear")
    y0 = signal.detrend(np.asarray(y, dtype=float), type="linear")
    scale = float(np.std(x0, ddof=1))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Conditioning signal has zero or invalid standard deviation")
    events = {}
    for label, condition in (("positive", x0 >= threshold_sigma * scale),
                             ("negative", x0 <= -threshold_sigma * scale)):
        candidates = np.flatnonzero(condition)
        keep, last = [], -10**18
        min_distance = max(1, half_window // 2)
        for idx in candidates:
            if idx - last >= min_distance and idx >= half_window and idx + half_window < len(x0):
                keep.append(int(idx)); last = int(idx)
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
                        help="Magnetic field in tesla; default 1.8 T is the paper-stated value near LCFS (override for selected radial position).")
    parser.add_argument("--separation-m", type=float, default=0.008,
                        help="vf2-vf3 probe separation (paper: 8 mm).")
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1,
                        help="Sign convention multiplier for E=(vf3-vf2)/separation.")
    parser.add_argument("--max-download-mb", type=float, default=200,
                        help="Safety limit per downloaded file.")
    args = parser.parse_args()
    if not math.isfinite(args.b_tesla) or args.b_tesla <= 0:
        parser.error("--b-tesla must be finite and > 0")
    if not math.isfinite(args.separation_m) or args.separation_m <= 0:
        parser.error("--separation-m must be finite and > 0")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    series, provenance = {}, {}
    for key, filename in FILES.items():
        url = f"{RECORD}/preview/{filename}?include_deleted=0"
        raw = fetch(url)
        if len(raw) > args.max_download_mb * 1024 * 1024:
            raise RuntimeError(f"{filename} exceeds --max-download-mb safety limit")
        (outdir / filename).write_bytes(raw)
        series[key] = parse_numeric_series(raw, filename)
        provenance[key] = {"url": url, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "samples": int(series[key].size),
            "min": float(np.min(series[key])), "max": float(np.max(series[key])),
            "mean": float(np.mean(series[key])), "std_sample": float(np.std(series[key], ddof=1))}
    lengths = {len(x) for x in series.values()}
    if len(lengths) != 1:
        raise ValueError(f"Signal lengths differ; no implicit truncation allowed: { {k: len(v) for k,v in series.items()} }")
    ne, vf2, vf3 = (series[k] for k in ("ne", "vf2", "vf3"))
    electric_field = args.sign * (vf3 - vf2) / args.separation_m
    flux = ne * electric_field / args.b_tesla
    flux_detrended = signal.detrend(flux, type="linear")
    sigma_flux = float(np.std(flux_detrended, ddof=1))
    relative_difference = abs(sigma_flux - PUBLISHED_SIGMA_FLUX) / PUBLISHED_SIGMA_FLUX * 100.0
    report = {
        "title": "EAST shot 22010: measured-series particle-flux statistics",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_record": RECORD, "paper_doi": PAPER_DOI,\n        "dataset_readme_text": readme_text,
        "method": {"density_series": FILES["ne"], "potential_series": [FILES["vf2"], FILES["vf3"]],
            "electric_field_formula": "E23=(vf3-vf2)/Delta", "particle_flux_formula": "F=ne*E23/B",
            "detrending": "linear detrend of constructed flux before standard deviation",
            "probe_separation_m": args.separation_m, "magnetic_field_T_user_supplied": args.b_tesla,
            "sign_multiplier": args.sign,
            "important_limitation": "Comparison is meaningful only if supplied B, file units, sign convention, and preprocessing match the paper. This script does not infer or guess them."},
        "source_files": provenance,
        "derived": {"samples": int(len(flux)),
            "electric_field_V_per_m_std": float(np.std(signal.detrend(electric_field), ddof=1)),
            "flux_mean_after_detrend_m-2_s-1": float(np.mean(flux_detrended)),
            "flux_sigma_after_detrend_m-2_s-1": sigma_flux,
            "published_flux_mean_m-2_s-1": PUBLISHED_MEAN_FLUX,
            "published_flux_sigma_m-2_s-1": PUBLISHED_SIGMA_FLUX,
            "mean_absolute_relative_difference_percent": abs(float(np.mean(flux_detrended)) - PUBLISHED_MEAN_FLUX) / abs(PUBLISHED_MEAN_FLUX) * 100.0,
            "absolute_relative_difference_percent": relative_difference,
            "flux_finite_fraction": float(np.mean(np.isfinite(flux)))},
        "conditional_averages": conditional_average(flux_detrended, flux_detrended),
        "acceptance": {"numeric_comparison_computed": True, "empirical_validation_pass": False,
            "reason": "Numeric difference is reported, but acceptance remains pending independent confirmation of exact B, unit scaling, detrending convention, and same-channel processing from paper/data documentation."}}
    (outdir / "east_22010_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (outdir / "east_22010_flux.csv").open("w", encoding="utf-8") as f:
        f.write("sample,ne,vf2,vf3,electric_field_V_m,flux_m-2_s-1,flux_detrended_m-2_s-1\n")
        for i in range(len(flux)):
            f.write(f"{i},{ne[i]:.12g},{vf2[i]:.12g},{vf3[i]:.12g},{electric_field[i]:.12g},{flux[i]:.12g},{flux_detrended[i]:.12g}\n")
    print(json.dumps({"report": str(outdir / "east_22010_validation.json"), "samples": len(flux),
        "sigma_flux_computed": sigma_flux, "sigma_flux_published": PUBLISHED_SIGMA_FLUX, "mean_flux_published": PUBLISHED_MEAN_FLUX,
        "relative_difference_percent": relative_difference, "empirical_validation_pass": False}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
