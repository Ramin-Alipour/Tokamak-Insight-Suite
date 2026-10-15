#!/usr/bin/env python3
"""Reproducible public-data spectral integrity test for GOLEM shot 30622.

This is a data-ingestion and spectral-shape test, not a claim of independent
Doppler-temperature validation. The GOLEM archive's temperature product is
labelled electron temperature, so it is not treated as ion temperature.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

SHOT = 30622
BASES = [
    # The GOLEM SpectrometerControl.py public example documents this HTTP endpoint.
    f"http://golem.fjfi.cvut.cz/operation/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
    f"https://golem.fjfi.cvut.cz/operation/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
    f"http://golem.fjfi.cvut.cz/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
    f"https://golem.fjfi.cvut.cz/shots/{SHOT}/diagnostics/Radiation/1111Spectrometer.ON/data/",
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

def get(url: str):
    req = Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-validation/1.0"})
    with urlopen(req, timeout=25) as r:
        return r.read()

def numeric_rows(blob: bytes):
    rows = []
    for line in blob.decode("utf-8", "replace").splitlines():
        s = line.strip().replace(",", " ")
        if not s or s.startswith(("#", "%", "//")):
            continue
        parts = s.split()
        vals = []
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if vals and all(__import__("math").isfinite(v) for v in vals):
            rows.append(vals)
    return rows

report = {
    "test": "GOLEM public spectroscopy data ingestion / integrity",
    "shot": SHOT,
    "source_page": f"https://golem.fjfi.cvut.cz/shots/{SHOT}/Data.php",
    "status": "FAIL",
    "files": {},
    "limitations": [
        "Archive temperature output is labelled electron temperature, not ion temperature.",
        "A spectral line fit is not a Doppler-temperature validation unless wavelength calibration, line identity, instrumental width, and ion mass are independently established.",
    ],
}
for key, filename in FILES.items():
    found = None
    errors = []
    for base in BASES:
        url = base + filename
        try:
            blob = get(url)
            if len(blob) < 16:
                errors.append(f"{url}: response too short")
                continue
            # Reject HTML error pages masquerading as data.
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
if raw.get("numeric_rows", 0) >= 10:
    report["status"] = "PASS_DATA_INGESTION"
    report["interpretation"] = (
        "Raw spectral file retrieved and numerically parsed. This confirms data access and "
        "basic integrity only; physics-level line fitting and independent temperature validation "
        "require a documented wavelength axis, line identification, instrument function, and "
        "comparable reference output."
    )
else:
    report["interpretation"] = (
        "Could not retrieve/parse a raw spectrum from the documented candidate paths. "
        "No spectral fit or temperature result is fabricated."
    )
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["status"] != "PASS_DATA_INGESTION":
    sys.exit(2)
