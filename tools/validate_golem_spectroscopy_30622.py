#!/usr/bin/env python3
"""Reproducible GOLEM shot 30622 spectral ingestion and Balmer-line shape audit.

The fitted widths combine plasma broadening, instrument response, and sampling.
This is not a Doppler-temperature result because the instrumental line-spread
function and a verified calibration uncertainty are not available here.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
from scipy.optimize import curve_fit

SHOT = 30622
BASES = [
    f"https://golem.fjfi.cvut.cz/operation/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
    f"http://golem.fjfi.cvut.cz/operation/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
    f"https://golem.fjfi.cvut.cz/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
    f"http://golem.fjfi.cvut.cz/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
]
FILES = {
    "raw_spectrum": "spectra.txt_Data_.txt",
    "calibrated_spectrum": "spectrum_calibrated",
    "residual_spectrum": "residual_spectra",
    "projection_basis": "components",
    "temperature_electron": "temperature",
}
OUT = Path("artifacts/golem-spectroscopy-30622")
OUT.mkdir(parents=True, exist_ok=True)

def get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-validation/1.1"})
    with urlopen(req, timeout=25) as response:
        return response.read()

def numeric_rows(blob: bytes):
    rows = []
    for line in blob.decode("utf-8", "replace").splitlines():
        s = line.strip().replace(",", " ")
        if not s or s.startswith(("#", "%", "//")):
            continue
        try:
            vals = [float(x) for x in s.split()]
        except ValueError:
            continue
        if vals and all(math.isfinite(v) for v in vals):
            rows.append(vals)
    return rows

def parse_spectrum(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            continue
        # wavelength + 29 acquisitions for this file
        if len(vals) == 30 and all(math.isfinite(v) for v in vals):
            rows.append(vals)
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 100 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected spectrum matrix shape: {arr.shape}")
    return arr[:, 0], arr[:, 1:]

def gaussian_linear(x, amplitude, center, sigma, intercept, slope):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + intercept + slope * (x - center)

def fit_line(wavelength, intensity, name, reference_nm, window):
    mask = (wavelength >= window[0]) & (wavelength <= window[1])
    x, y = wavelength[mask], intensity[mask]
    if len(x) < 8:
        raise ValueError(f"Too few spectral points for {name}")
    baseline = float(np.percentile(y, 20))
    center0 = float(x[int(np.argmax(y))])
    p0 = [max(float(np.max(y)) - baseline, 1.0), center0, 0.7, baseline, 0.0]
    params, _ = curve_fit(
        gaussian_linear, x, y, p0=p0, maxfev=30000,
        bounds=([0, window[0], 0.08, -1e5, -1e4],
                [1e7, window[1], 8.0, 1e5, 1e4]),
    )
    amp, center, sigma, intercept, slope = [float(v) for v in params]
    fitted = gaussian_linear(x, *params)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "reference_air_wavelength_nm_approx": reference_nm,
        "fitted_center_nm": center,
        "center_offset_from_reference_nm": center - reference_nm,
        "gaussian_fwhm_nm": 2.354820045 * sigma,
        "gaussian_amplitude_counts": amp,
        "fitted_peak_counts": amp + intercept,
        "fit_r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "fit_window_nm": list(window),
    }

report = {
    "test": "GOLEM shot 30622 spectral ingestion and Balmer-line shape audit",
    "shot": SHOT,
    "source_page": f"https://golem.fjfi.cvut.cz/shots/{SHOT}/Data.php",
    "status": "FAIL",
    "files": {},
    "spectral_line_audit": {},
    "limitations": [
        "The line widths include instrumental broadening and are not ion-temperature measurements.",
        "No instrument line-spread function or independently verified wavelength-calibration uncertainty was available in this run.",
        "The GOLEM archive temperature product is not assumed to be ion temperature.",
        "A line-center comparison is a coarse identification/calibration diagnostic, not a full independent calibration.",
    ],
}
for key, filename in FILES.items():
    found, errors = None, []
    for base in BASES:
        url = base + filename
        try:
            blob = get(url)
            if len(blob) < 16:
                errors.append(f"{url}: response too short")
                continue
            prefix = blob[:500].lower()
            if b"<html" in prefix or b"<!doctype" in prefix:
                errors.append(f"{url}: HTML response, not data")
                continue
            found = (url, blob)
            break
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    if found:
        url, blob = found
        path = OUT / f"{key}.txt"
        path.write_bytes(blob)
        rows = numeric_rows(blob)
        report["files"][key] = {
            "url": url, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
            "numeric_rows": len(rows), "max_columns": max((len(r) for r in rows), default=0),
            "saved_as": str(path),
        }
    else:
        report["files"][key] = {"status": "unavailable", "attempts": errors}

raw = report["files"].get("raw_spectrum", {})
if raw.get("numeric_rows", 0) >= 100:
    try:
        wavelength, spectra = parse_spectrum(OUT / "raw_spectrum.txt")
        mean_spectrum = np.mean(spectra, axis=1)
        lines = {
            "H_beta": (486.135, (480.0, 492.0)),
            "H_alpha": (656.281, (650.0, 662.0)),
            "H_gamma": (434.047, (429.0, 439.0)),
        }
        for name, (reference, window) in lines.items():
            report["spectral_line_audit"][name] = fit_line(
                wavelength, mean_spectrum, name, reference, window
            )
        report["spectrum_matrix"] = {
            "wavelength_samples": int(len(wavelength)),
            "acquisitions": int(spectra.shape[1]),
            "wavelength_min_nm": float(np.min(wavelength)),
            "wavelength_max_nm": float(np.max(wavelength)),
            "median_sampling_interval_nm": float(np.median(np.diff(wavelength))),
            "noise_rms_counts_header": 10.6,
            "analysis": "Gaussian plus linear-background fits to the mean of all recorded acquisitions",
        }
        # Time-resolved audit: reject noise-dominated acquisitions instead of
        # treating a numerically converged fit as a detected line.
        time_header = next((line for line in (OUT / "raw_spectrum.txt").read_text(encoding="utf-8", errors="replace").splitlines()
                            if line.startswith("Time stamps [ms]:")), "")
        import re
        time_match = re.search(r"\[(.*?)\]", time_header)
        timestamps = [float(v) for v in time_match.group(1).split()] if time_match else []
        temporal = {}
        for line_name, (reference, window) in lines.items():
            mask = (wavelength >= window[0]) & (wavelength <= window[1])
            x = wavelength[mask]
            records = []
            for index in range(spectra.shape[1]):
                y = spectra[mask, index]
                baseline = float(np.percentile(y, 20))
                center0 = float(x[int(np.argmax(y))])
                try:
                    pars, _ = curve_fit(
                        gaussian_linear, x, y,
                        p0=[max(float(np.max(y)) - baseline, 1.0), center0, 0.7, baseline, 0.0],
                        maxfev=30000,
                        bounds=([0, window[0], 0.08, -1e5, -1e4],
                                [1e7, window[1], 8.0, 1e5, 1e4]),
                    )
                    amp, center, sigma, intercept, slope = [float(v) for v in pars]
                    fitted = gaussian_linear(x, *pars)
                    ss_res = float(np.sum((y - fitted) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
                    records.append({
                        "acquisition": index + 1,
                        "timestamp_ms": timestamps[index] if index < len(timestamps) else None,
                        "center_nm": center,
                        "fwhm_nm": 2.354820045 * sigma,
                        "peak_counts": amp + intercept,
                        "r_squared": r2,
                        "accepted": bool((amp + intercept) >= 500.0 and r2 is not None and r2 >= 0.90),
                    })
                except Exception as exc:
                    records.append({"acquisition": index + 1, "fit_error": type(exc).__name__, "accepted": False})
            good = [r for r in records if r.get("accepted")]
            centers = np.array([r["center_nm"] for r in good], dtype=float)
            widths = np.array([r["fwhm_nm"] for r in good], dtype=float)
            temporal[line_name] = {
                "selection_rule": "fitted peak >= 500 counts and R-squared >= 0.90",
                "accepted_acquisitions": len(good),
                "total_acquisitions": len(records),
                "center_mean_nm": float(np.mean(centers)) if len(centers) else None,
                "center_sample_std_nm": float(np.std(centers, ddof=1)) if len(centers) > 1 else None,
                "center_min_nm": float(np.min(centers)) if len(centers) else None,
                "center_max_nm": float(np.max(centers)) if len(centers) else None,
                "fwhm_mean_nm": float(np.mean(widths)) if len(widths) else None,
                "fwhm_sample_std_nm": float(np.std(widths, ddof=1)) if len(widths) > 1 else None,
                "accepted_records": good,
                "all_records": records,
            }
        report["time_resolved_line_audit"] = temporal
        report["status"] = "PASS_SPECTRAL_LINE_SHAPE"
        report["interpretation"] = (
            "Raw spectrum was retrieved and three Balmer-region line profiles were fitted. "
            "This demonstrates a reproducible spectral-shape analysis path; it does not validate "
            "Doppler temperature because instrument broadening and calibration uncertainty remain unresolved."
        )
    except Exception as exc:
        report["status"] = "PASS_DATA_INGESTION_ONLY"
        report["spectral_analysis_error"] = f"{type(exc).__name__}: {exc}"
        report["interpretation"] = "Raw spectrum was retrieved, but the line-shape analysis did not complete."
else:
    report["interpretation"] = "Raw spectrum could not be retrieved and parsed; no spectral result is fabricated."

(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["status"] != "PASS_SPECTRAL_LINE_SHAPE":
    sys.exit(2)
