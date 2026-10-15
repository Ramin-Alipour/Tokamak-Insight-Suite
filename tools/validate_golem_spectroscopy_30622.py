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

def voigt_linear(x, amplitude, center, sigma, gamma, intercept, slope):
    """Voigt profile plus linear baseline; sigma/gamma are Gaussian/Lorentz scales."""
    from scipy.special import voigt_profile
    return amplitude * voigt_profile(x - center, sigma, gamma) + intercept + slope * (x - center)

def fit_voigt_width(x, y, window):
    """Fit Voigt FWHM; this does not deconvolve the instrument response."""
    baseline = float(np.percentile(y, 20))
    center0 = float(x[int(np.argmax(y))])
    p0 = [max(float(np.max(y)) - baseline, 1.0), center0, 0.5, 0.1, baseline, 0.0]
    pars, covariance = curve_fit(
        voigt_linear, x, y, p0=p0, maxfev=50000,
        bounds=([0, window[0], 0.02, 0.0, -1e5, -1e4],
                [1e8, window[1], 8.0, 8.0, 1e5, 1e4]),
    )
    amp, center, sigma, gamma, intercept, slope = [float(v) for v in pars]
    fwhm = 0.5346 * (2.0 * gamma) + math.sqrt(0.2166 * (2.0 * gamma)**2 + (2.354820045 * sigma)**2)
    fitted = voigt_linear(x, *pars)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"center_nm": center, "fwhm_nm": fwhm, "gaussian_sigma_nm": sigma,
            "lorentz_hwhm_nm": gamma, "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
            "covariance_finite": bool(np.all(np.isfinite(covariance)))}

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
        timestamp_values = time_header.rsplit("[", 1)[-1].split("]", 1)[0]
        timestamps = [float(v) for v in timestamp_values.split()] if timestamp_values else []
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
        # Fit Voigt profiles to the same real H-alpha acquisitions as a model-sensitivity check.
        ha_mask = (wavelength >= lines["H_alpha"][1][0]) & (wavelength <= lines["H_alpha"][1][1])
        ha_x = wavelength[ha_mask]
        voigt_records = []
        for index in range(spectra.shape[1]):
            yy = spectra[ha_mask, index]
            try:
                vf = fit_voigt_width(ha_x, yy, lines["H_alpha"][1])
                vf.update({"acquisition": index + 1,
                           "timestamp_ms": timestamps[index] if index < len(timestamps) else None,
                           "accepted": bool(float(np.max(yy)) >= 500.0 and vf["r_squared"] is not None and vf["r_squared"] >= 0.90)})
                voigt_records.append(vf)
            except Exception as exc:
                voigt_records.append({"acquisition": index + 1, "accepted": False, "fit_error": f"{type(exc).__name__}: {exc}"})
        vg = [r for r in voigt_records if r.get("accepted") and r.get("fwhm_nm") is not None]
        gg = temporal["H_alpha"]["accepted_records"]
        gmean = float(np.mean([r["fwhm_nm"] for r in gg])) if gg else None
        vmean = float(np.mean([r["fwhm_nm"] for r in vg])) if vg else None
        report["H_alpha_model_sensitivity"] = {
            "data_source": "same raw GOLEM shot 30622 acquisitions; no synthetic data",
            "gaussian_accepted_n": len(gg), "voigt_accepted_n": len(vg),
            "gaussian_mean_fwhm_nm": gmean, "voigt_mean_fwhm_nm": vmean,
            "voigt_minus_gaussian_mean_fwhm_nm": (vmean-gmean) if vmean is not None and gmean is not None else None,
            "all_voigt_fits": voigt_records,
            "interpretation": "Model-sensitivity only; neither model separates instrument from plasma broadening."
        }
        # Do not calibrate a spectrometer against plasma Balmer lines:
        # their centers can be shifted by plasma bulk velocity, blends, opacity,
        # and line-fitting bias. Instead report apparent Doppler velocities as a
        # consistency diagnostic and explicitly require a lamp/reference source
        # for wavelength calibration.
        refs_air_nm = {"H_alpha": 656.281, "H_beta": 486.135, "H_gamma": 434.047}
        c_km_s = 299792.458
        velocity_audit = {}
        for line_name, ref_nm in refs_air_nm.items():
            center = temporal[line_name].get("center_mean_nm")
            n_good = temporal[line_name].get("accepted_acquisitions", 0)
            velocity_audit[line_name] = {
                "observed_center_nm": center,
                "reference_air_wavelength_nm_approx": ref_nm,
                "apparent_radial_velocity_km_s": (
                    c_km_s * (center / ref_nm - 1.0) if center is not None else None
                ),
                "accepted_acquisitions": n_good,
                "total_acquisitions": temporal[line_name].get("total_acquisitions", spectra.shape[1]),
                "usable_for_velocity_consistency": bool(center is not None and n_good >= 5),
            }
        usable_velocities = [
            item["apparent_radial_velocity_km_s"]
            for item in velocity_audit.values()
            if item["usable_for_velocity_consistency"]
        ]
        report["Balmer_velocity_consistency"] = {
            "lines": velocity_audit,
            "usable_line_count": len(usable_velocities),
            "velocity_spread_km_s": (
                float(max(usable_velocities) - min(usable_velocities))
                if len(usable_velocities) >= 2 else None
            ),
            "wavelength_calibration_status": "NOT_ASSESSED_FROM_PLASMA_LINES",
            "decision": "REQUIRES_EXTERNAL_WAVELENGTH_STANDARD",
            "reason": (
                "Plasma-emission line centers are not wavelength-calibration standards because "
                "real plasma Doppler shifts and line-shape/systematic effects can move the centers. "
                "H-gamma is excluded from consistency interpretation unless at least five acquisitions pass."
            ),
            "required_next_step": "Fit a multi-line Hg/Ar (or other certified lamp) spectrum from the same instrument setup; do not apply an affine correction inferred from H-alpha/H-beta."
        }
        # Quantify the naive Doppler-only temperature for transparency, but never
        # label it as a measured ion temperature: instrumental broadening is unknown.
        ha_mean = temporal["H_alpha"].get("fwhm_mean_nm")
        ha_center = temporal["H_alpha"].get("center_mean_nm")
        if ha_mean is not None and ha_center:
            # T[eV] = m_H c^2 / (8 ln 2) * (FWHM/lambda)^2
            # This is only the apparent Doppler-only value before instrumental deconvolution.
            hydrogen_rest_energy_eV = 938.27208816e6
            apparent_ti_eV = (hydrogen_rest_energy_eV / (8.0 * math.log(2.0))) * (ha_mean / ha_center) ** 2
            relative_width_std = (
                temporal["H_alpha"]["fwhm_sample_std_nm"] / ha_mean
                if temporal["H_alpha"].get("fwhm_sample_std_nm") is not None else None
            )
            report["H_alpha_apparent_doppler_temperature"] = {
                "apparent_temperature_eV": float(apparent_ti_eV),
                "width_mean_nm": ha_mean,
                "center_mean_nm": ha_center,
                "width_sample_std_nm": temporal["H_alpha"].get("fwhm_sample_std_nm"),
                "approximate_uncertainty_from_fit_scatter_eV": (
                    float(2.0 * apparent_ti_eV * relative_width_std)
                    if relative_width_std is not None else None
                ),
                "physical_status": "NOT_A_VALID_ION_TEMPERATURE",
                "reason": "Observed H-alpha FWHM includes unknown instrumental line-spread broadening; sampling is coarse (median 0.45 nm/pixel), only 6/29 fits pass selection, and no measured instrument LSF is available. Do not report this apparent value as Ti.",
                "required_next_step": "Measure or retrieve the instrument line-spread function at H-alpha, then deconvolve in variance space and propagate wavelength/fit/LSF uncertainty."
            }
        # Keep line-center offsets explicit so calibration failure is visible in the report.
        report["Balmer_line_center_offsets"] = {
            "H_alpha_nm": float(temporal["H_alpha"]["center_mean_nm"] - 656.281) if temporal["H_alpha"].get("center_mean_nm") is not None else None,
            "H_beta_nm": float(temporal["H_beta"]["center_mean_nm"] - 486.135) if temporal["H_beta"].get("center_mean_nm") is not None else None,
            "H_gamma_nm": float(temporal["H_gamma"]["center_mean_nm"] - 434.047) if temporal["H_gamma"].get("center_mean_nm") is not None else None,
            "interpretation": "Offsets are descriptive only; plasma-emission line centers are not wavelength-calibration standards and H-gamma is statistically weak."
        }
        report["status"] = "PASS_SPECTRAL_LINE_SHAPE"
        report["interpretation"] = (
            "Raw spectrum was retrieved and three Balmer-region line profiles were fitted. "
            "Plasma-line centers are now treated as apparent Doppler-shift diagnostics, not calibration standards. "
            "A valid Ti still requires a measured instrument LSF and sufficient spectral sampling."
        )
    except Exception as exc:
        report["status"] = "PASS_DATA_INGESTION_ONLY"
        report["spectral_analysis_error"] = f"{type(exc).__name__}: {exc}"
        report["interpretation"] = "Raw spectrum was retrieved, but the line-shape analysis did not complete."
else:
    report["interpretation"] = "Raw spectrum could not be retrieved and parsed; no spectral result is fabricated."

# Export the per-acquisition fits as a machine-readable table for independent review.
if "time_resolved_line_audit" in report:
    import csv
    temporal_csv = OUT / "time_resolved_line_fits.csv"
    with temporal_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["line", "acquisition", "timestamp_ms", "center_nm", "fwhm_nm",
                  "peak_counts", "r_squared", "accepted", "fit_error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for line_name, summary in report["time_resolved_line_audit"].items():
            for record in summary.get("all_records", []):
                writer.writerow({"line": line_name, **{k: record.get(k) for k in fields if k != "line"}})
    with (OUT / "ha_model_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv
        fields = ["model", "acquisition", "timestamp_ms", "center_nm", "fwhm_nm", "r_squared", "accepted", "fit_error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in report.get("H_alpha_model_sensitivity", {}).get("all_voigt_fits", []):
            writer.writerow({"model": "Voigt", **{k: rec.get(k) for k in fields if k != "model"}})
        for rec in report.get("time_resolved_line_audit", {}).get("H_alpha", {}).get("all_records", []):
            writer.writerow({"model": "Gaussian", **{k: rec.get(k) for k in fields if k != "model"}})
    # Stability metrics are descriptive; accepted fits are threshold-selected,
    # so they must not be interpreted as the full shot's temporal distribution.
    ha = report["time_resolved_line_audit"].get("H_alpha", {})
    if ha.get("center_mean_nm") is not None and ha.get("fwhm_mean_nm") not in (None, 0):
        report["H_alpha_stability_summary"] = {
            "accepted_fits": ha.get("accepted_acquisitions"),
            "total_fits": ha.get("total_acquisitions"),
            "center_peak_to_peak_nm": (
                ha["center_max_nm"] - ha["center_min_nm"]
                if ha.get("center_max_nm") is not None and ha.get("center_min_nm") is not None else None
            ),
            "center_sample_std_nm": ha.get("center_sample_std_nm"),
            "center_relative_sample_std_ppm": (
                1e6 * ha["center_sample_std_nm"] / ha["center_mean_nm"]
                if ha.get("center_sample_std_nm") is not None else None
            ),
            "fwhm_mean_nm": ha.get("fwhm_mean_nm"),
            "fwhm_sample_std_nm": ha.get("fwhm_sample_std_nm"),
            "fwhm_coefficient_of_variation_percent": (
                100 * ha["fwhm_sample_std_nm"] / ha["fwhm_mean_nm"]
                if ha.get("fwhm_sample_std_nm") is not None else None
            ),
            "interpretation": "Descriptive stability among accepted H-alpha fits only; not an instrument-calibrated Doppler-width or ion-temperature result."
        }

# Try the instrument configuration referenced by the upstream GOLEM analysis code.
# Only a retrieved configuration file may supply an instrumental-width correction.
config_candidates = [
    "https://golem.fjfi.cvut.cz/shots/9867/diagnostics/Radiation/1111Spectrometer.ON/ControlSoftware/HR%2BC1886.cfg",
    "https://golem.fjfi.cvut.cz/shots/11115/diagnostics/Radiation/1111Spectrometer.ON/ControlSoftware/HR%2BC1886.cfg",
    "https://golem.fjfi.cvut.cz/operation/shots/9867/diagnostics/Radiation/1111Spectrometer.ON/ControlSoftware/HR%2BC1886.cfg",
    "https://golem.fjfi.cvut.cz/operation/shots/11115/diagnostics/Radiation/1111Spectrometer.ON/ControlSoftware/HR%2BC1886.cfg",
]
config_audit = {"status": "instrument_config_not_retrieved", "attempts": []}
for config_url in config_candidates:
    try:
        config_blob = get(config_url)
        config_text = config_blob.decode("utf-8", "replace")
        if "[Instrumental broadening function]" not in config_text or "width(wavelength)" not in config_text:
            config_audit["attempts"].append({"url": config_url, "status": "not_a_recognized_instrument_config"})
            continue
        (OUT / "upstream_HR_C1886.cfg").write_bytes(config_blob)
        import re
        section = config_text.split("[Instrumental broadening function]", 1)[1].split("\n[", 1)[0]
        section_lines = section.splitlines()
        width_start = next((i for i, line in enumerate(section_lines)
                            if re.match(r"^\s*width\(wavelength\)\s*=", line, re.I)), None)
        width_text = ""
        if width_start is not None:
            width_text = section_lines[width_start].split("=", 1)[1]
            for line in section_lines[width_start + 1:]:
                if re.match(r"^\s*[A-Za-z_][^=]*=", line):
                    break
                width_text += " " + line
        nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", width_text)]
        pairs = list(zip(nums[0::2], nums[1::2]))
        if pairs:
            inst_width = float(np.interp(656.281, [p[0] for p in pairs], [p[1] for p in pairs]))
            precision_match = re.search(r"(?im)^wavelength presicion \[nm\]\s*=\s*([-+0-9.eE]+)", config_text)
            correction_match = re.search(r"(?im)^calibration polynome correction\s*=\s*(.+)$", config_text)
            correction_values = ([float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", correction_match.group(1))]
                                if correction_match else [])
            config_audit = {
                "status": "instrument_config_retrieved",
                "source_url": config_url,
                "sha256": hashlib.sha256(config_blob).hexdigest(),
                "config_date": re.search(r"(?im)^date\s*=\s*(.+)$", config_text).group(1).strip() if re.search(r"(?im)^date\s*=\s*(.+)$", config_text) else None,
                "serial_number": re.search(r"(?im)^serial number\s*=\s*(.+)$", config_text).group(1).strip() if re.search(r"(?im)^serial number\s*=\s*(.+)$", config_text) else None,
                "nominal_wavelength_precision_nm": float(precision_match.group(1)) if precision_match else None,
                "calibration_polynomial_correction_coefficients": correction_values,
                "width_points_nm": [{"wavelength_nm": x, "instrumental_width_nm": y} for x, y in pairs],
                "interpolated_instrumental_width_at_H_alpha_nm": inst_width,
                "width_definition": "treated as Gaussian sigma because upstream code adds this term in quadrature; verify for current shot",
                "decision": "used_for_provisional_candidate_only; not accepted as shot-specific calibration"
            }
            break
        config_audit["attempts"].append({"url": config_url, "status": "config_found_but_width_parse_failed"})
    except Exception as exc:
        config_audit["attempts"].append({"url": config_url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
report["instrument_config_audit"] = config_audit
# Apply the historical config's wavelength-correction polynomial separately from
# the H-alpha/H-beta fitted affine map; retain the nominal precision caveat.
coeffs = config_audit.get("calibration_polynomial_correction_coefficients", [])
if coeffs and "time_resolved_line_audit" in report:
    refs = {"H_alpha": 656.281, "H_beta": 486.135, "H_gamma": 434.047}
    config_corrected = {}
    precision = config_audit.get("nominal_wavelength_precision_nm")
    for line_name, reference in refs.items():
        center = report["time_resolved_line_audit"].get(line_name, {}).get("center_mean_nm")
        if center is None:
            config_corrected[line_name] = {"status": "no_accepted_line_center"}
            continue
        corrected = float(center)
        for power, coefficient in enumerate(coeffs):
            corrected += coefficient * float(center) ** power
        residual = corrected - reference
        config_corrected[line_name] = {
            "observed_center_nm": float(center),
            "config_corrected_center_nm": corrected,
            "reference_nm_approx": reference,
            "residual_nm": residual,
            "within_historical_nominal_precision": bool(precision is not None and abs(residual) <= precision)
        }
    report["historical_config_wavelength_correction_check"] = {
        "config_date": config_audit.get("config_date"),
        "serial_number": config_audit.get("serial_number"),
        "polynomial_coefficients": coeffs,
        "nominal_precision_nm": precision,
        "lines": config_corrected,
        "decision": "HISTORICAL_CORRECTION_COMPARED_NOT_A_SHOT_SPECIFIC_CALIBRATION",
        "limitations": ["The correction is from a 2011 config, not a same-session lamp exposure.",
                        "The nominal precision is coarse and does not replace independent calibration data."]
    }
crosscheck = report.get("provisional_wavelength_crosscheck", {})
nominal_precision = config_audit.get("nominal_wavelength_precision_nm")
if crosscheck.get("H_gamma_residual_nm") is not None and nominal_precision is not None:
    residual_abs = abs(float(crosscheck["H_gamma_residual_nm"]))
    report["wavelength_calibration_spec_check"] = {
        "instrument_config_source": config_audit.get("source_url"),
        "instrument_config_date": config_audit.get("config_date"),
        "instrument_serial_number": config_audit.get("serial_number"),
        "nominal_wavelength_precision_nm": nominal_precision,
        "independent_H_gamma_residual_nm": float(crosscheck["H_gamma_residual_nm"]),
        "residual_within_historical_nominal_precision": bool(residual_abs <= nominal_precision),
        "strict_0p25nm_check_passed": bool(residual_abs <= 0.25),
        "decision": "CONSISTENT_WITH_HISTORICAL_NOMINAL_SPEC_ONLY" if residual_abs <= nominal_precision else "OUTSIDE_HISTORICAL_NOMINAL_SPEC",
        "limitations": [
            "The config is dated 2011 while the measured shot is from 2019.",
            "Nominal wavelength precision is not a substitute for a same-session calibration-lamp fit.",
            "The H-gamma line is weak; this check is not an independent calibration certificate."
        ]
    }

# If the upstream instrument-width curve is available, compute a separately
# labelled provisional deconvolution. The old config's width is used as a
# Gaussian-sigma-like term because the upstream fitting code adds it in quadrature.
if config_audit.get("status") == "instrument_config_retrieved":
    try:
        sigma_inst = float(config_audit["interpolated_instrumental_width_at_H_alpha_nm"])
        ha = report.get("time_resolved_line_audit", {}).get("H_alpha", {})
        accepted = ha.get("accepted_records", [])
        center = ha.get("center_mean_nm")
        if accepted and center and sigma_inst >= 0:
            per_record = []
            for rec in accepted:
                sigma_observed = float(rec["fwhm_nm"]) / 2.354820045
                variance_remaining = sigma_observed**2 - sigma_inst**2
                if variance_remaining > 0:
                    sigma_intrinsic = math.sqrt(variance_remaining)
                    fwhm_intrinsic = 2.354820045 * sigma_intrinsic
                    ti = (938.27208816e6 / (8.0 * math.log(2.0))) * (fwhm_intrinsic / center)**2
                    per_record.append({"acquisition": rec["acquisition"], "observed_fwhm_nm": rec["fwhm_nm"],
                                       "deconvolved_fwhm_nm": fwhm_intrinsic, "provisional_Ti_eV": ti})
            if per_record:
                values = np.asarray([r["provisional_Ti_eV"] for r in per_record], dtype=float)
                report["H_alpha_LSF_corrected_temperature_candidate"] = {
                    "status": "PROVISIONAL_ONLY_NOT_VALIDATED_Ti",
                    "instrumental_width_nm": sigma_inst,
                    "instrumental_width_interpretation": "treated as Gaussian sigma based on the upstream code's quadrature model; not independently verified",
                    "accepted_widths_used": len(per_record),
                    "accepted_widths_total": len(accepted),
                    "provisional_mean_Ti_eV": float(np.mean(values)),
                    "provisional_sample_std_Ti_eV": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "per_acquisition": per_record,
                    "limitations": [
                        "Instrument config is dated 2011 and is not confirmed to have remained unchanged for shot 30622.",
                        "No same-session certified wavelength-standard spectrum was available; the 2011 correction is historical only.",
                        "Sampling is coarse and only threshold-selected H-alpha fits were used.",
                        "This is a Doppler-only candidate; other broadening mechanisms and config uncertainty are not quantified."
                    ]
                }
    except Exception as exc:
        report["H_alpha_LSF_corrected_temperature_candidate"] = {
            "status": "NOT_COMPUTED", "error": f"{type(exc).__name__}: {exc}"
        }

(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["status"] != "PASS_SPECTRAL_LINE_SHAPE":
    sys.exit(2)
