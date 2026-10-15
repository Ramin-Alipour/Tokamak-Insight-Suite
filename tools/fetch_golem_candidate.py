#!/usr/bin/env python3
"""Acquire GOLEM spectrum and temperature candidates with reproducible provenance.

Acquisition is not scientific validation: the script never labels a candidate as
raw or treats a published summary value as a matched reference automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from statistics import median
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

BASE = "https://golem.fjfi.cvut.cz"
API_BASE = "http://golem.fjfi.cvut.cz"
USER_AGENT = "Tokamak-Insight-Suite empirical-validation inventory/1.1"


def get(url: str) -> tuple[bytes, str, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
        return data, response.geturl(), content_type


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze_hydrogen_lines(path: Path) -> dict:
    """Check coarse Balmer-line identification against NIST air wavelengths.

    This deliberately does not estimate Doppler temperature: the wavelength
    sampling and unknown instrument line-spread function are insufficient.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in lines[11:]:
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            continue
        if len(values) == 30:  # wavelength + 29 individual exposures
            rows.append(values)
    if len(rows) < 100:
        return {"status": "not_evaluated", "reason": "Expected wavelength plus 29 spectra table was not recognized."}
    wavelength = [row[0] for row in rows]
    mean_counts = [sum(row[1:]) / len(row[1:]) for row in rows]
    spacing = float(median([wavelength[i + 1] - wavelength[i] for i in range(len(wavelength) - 1)]))
    references = [
        ("H_alpha", 656.281, 654.0, 660.0),
        ("H_beta", 486.134, 484.0, 489.0),
        ("H_gamma", 434.047, 431.0, 438.0),
    ]
    measured = []
    for name, reference_nm, low_nm, high_nm in references:
        selected = [
            (x, y) for x, y in zip(wavelength, mean_counts)
            if low_nm <= x <= high_nm
        ]
        if len(selected) < 3:
            measured.append({"line": name, "status": "not_evaluated", "reason": "Insufficient samples in line window."})
            continue
        x = [item[0] for item in selected]
        y = [item[1] for item in selected]
        baseline = _percentile(y, 0.20)
        weights = [max(value - baseline, 0.0) for value in y]
        if sum(weights) <= 0:
            measured.append({"line": name, "status": "not_evaluated", "reason": "No positive baseline-subtracted signal."})
            continue
        centroid_nm = sum(xi * wi for xi, wi in zip(x, weights)) / sum(weights)
        peak_index = max(range(len(y)), key=lambda i: y[i])
        half_max = baseline + (max(y) - baseline) / 2.0
        half_indices = [i for i, value in enumerate(y) if value >= half_max]
        fwhm_nm = x[half_indices[-1]] - x[half_indices[0]] if len(half_indices) >= 2 else None
        offset_nm = centroid_nm - reference_nm
        measured.append({
            "line": name,
            "reference_air_wavelength_nm": reference_nm,
            "measured_centroid_nm": centroid_nm,
            "measured_peak_nm": x[peak_index],
            "centroid_offset_nm": offset_nm,
            "centroid_offset_in_wavelength_bins": offset_nm / spacing,
            "observed_fwhm_nm": fwhm_nm,
            "within_one_wavelength_bin": abs(offset_nm) <= spacing,
            "method": "mean of 29 exposures; 20th-percentile local baseline; positive-weight centroid",
        })
    evaluated = [item for item in measured if item.get("status") != "not_evaluated"]
    passed = len(evaluated) == 3 and all(item["within_one_wavelength_bin"] for item in evaluated)
    return {
        "status": "PASS_coarse_line_identification" if passed else "FAIL_or_incomplete_coarse_line_identification",
        "reference_source": "NIST Handbook of Basic Atomic Spectroscopic Data, hydrogen strong lines",
        "reference_url": "https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable2.htm",
        "wavelength_bin_spacing_nm": spacing,
        "wavelength_bin_spacing_is_not_instrument_resolution": True,
        "lines": measured,
        "limitation": "Coarse line identification only. The unknown instrument line-spread function, sparse wavelength sampling, and unestablished calibration prevent Doppler-width or ion-temperature validation.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=30622)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/golem-empirical-candidate"))
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    shot = args.shot
    page_url = f"{BASE}/shots/{shot}/Data.php"
    page, final_page_url, page_type = get(page_url)
    (out / "shot_page.html").write_bytes(page)

    # The GOLEM Data page displays filenames, while its documented data API
    # accepts identifiers under /utils/data/<shot>/<identifier>[.txt].
    # Try both documented identifiers and legacy operation-directory paths.
    candidates = [
        ("spectrum_identifier", f"{API_BASE}/utils/data/{shot}/spectra.txt_Data_.txt"),
        ("spectrum_identifier_no_extension", f"{API_BASE}/utils/data/{shot}/spectra.txt_Data_"),
        ("spectrum_config_identifier", f"{API_BASE}/utils/data/{shot}/spectrometr:data"),
        ("calibrated_spectrum_identifier", f"{API_BASE}/utils/data/{shot}/spectrometr:data_calib"),
        ("calibrated_spectrum_database_path", f"{API_BASE}/golem/database/operation/shots/{shot}/diagnostics/Radiation/1111Spectrometer.ON/data/spectrum_calibrated"),
        ("spectrum_legacy_path", f"{BASE}/operation/shots/{shot}/diagnostics/Radiation/1111Spectrometer.ON/data/spectra.txt_Data_.txt"),
        ("impurity_temperature_identifier", f"{API_BASE}/utils/data/{shot}/spectrometr:temperature"),
        ("impurity_temperature_xls", f"{API_BASE}/utils/data/{shot}/spectrometr:temperature.xls"),
        ("impurity_temperature_database_path", f"{API_BASE}/golem/database/operation/shots/{shot}/analysis/Radiation/1212Impurities_TO.ON/data/temperature"),
        ("electron_temperature_mean_database_path", f"{API_BASE}/golem/database/operation/shots/{shot}/basicdiagn/ElectronTemperatureMean"),
        ("temperature_identifier", f"{API_BASE}/utils/data/{shot}/temperature"),
        ("temperature_mean_identifier", f"{API_BASE}/utils/data/{shot}/temperature_mean"),
        ("temperature_legacy_path", f"{BASE}/operation/shots/{shot}/temperature"),
        ("temperature_mean_legacy_path", f"{BASE}/operation/shots/{shot}/temperature_mean"),
    ]
    attempts, downloaded = [], []
    seen = set()
    for label, url in candidates:
        if url in seen:
            continue
        seen.add(url)
        record = {"label": label, "url": url}
        try:
            data, final_url, content_type = get(url)
            # Avoid mistaking an HTML error page served with status 200 for data.
            if "text/html" in content_type.lower() and b"<html" in data[:1000].lower():
                record.update(status="rejected_html", final_url=final_url, content_type=content_type,
                              bytes=len(data), reason="HTML response is not a data file")
            elif len(data) < 8:
                record.update(status="rejected_too_small", final_url=final_url, bytes=len(data))
            else:
                suffix = Path(urlparse(final_url).path).suffix or ".txt"
                filename = f"{label}{suffix}"
                (out / filename).write_bytes(data)
                record.update(status="downloaded", final_url=final_url, content_type=content_type,
                              file=filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                downloaded.append(record.copy())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        attempts.append(record)

    spectral_file = next(
        (out / item["file"] for item in downloaded
         if item["label"] == "spectrum_legacy_path" and (out / item["file"]).exists()),
        None,
    )
    spectral_line_check = analyze_hydrogen_lines(spectral_file) if spectral_file else {
        "status": "not_evaluated",
        "reason": "No recognized spectrum file was downloaded.",
    }

    manifest = {
        "shot": shot,
        "spectral_line_check": spectral_line_check,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "page_url": page_url,
        "final_page_url": final_page_url,
        "page_content_type": page_type,
        "page_bytes": len(page),
        "page_sha256": hashlib.sha256(page).hexdigest(),
        "attempts": attempts,
        "downloaded_files": downloaded,
        "public_page_reference_temperature_eV": 20.8 if shot == 30622 else None,
        "classification": "candidate data with a coarse Balmer-line identification check; no Doppler-width or temperature validation claimed",
        "acceptance_checks": [
            "Confirm downloaded file format and whether the spectrum is raw counts or corrected/calibrated data.",
            "Check coarse H-alpha, H-beta and H-gamma line identification against NIST wavelengths; do not interpret this as Doppler-width validation.",
            "Establish temporal/spatial correspondence between spectrum and the published temperature reference.",
            "Independently derive the same observable with documented uncertainty and calculate absolute/relative discrepancy.",
            "Add the verified input and numerical comparison as a deterministic regression test before declaring validation.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not downloaded:
        raise SystemExit("No candidate data files downloaded; this run is not an empirical-validation pass.")


if __name__ == "__main__":
    main()
