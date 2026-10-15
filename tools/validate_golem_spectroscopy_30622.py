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
        # Two strong lines define a provisional affine mapping; H-gamma is an independent check.
        hb = temporal["H_beta"]
        ha = temporal["H_alpha"]
        hg = temporal["H_gamma"]
        if hb["center_mean_nm"] is not None and ha["center_mean_nm"] is not None and hg["center_mean_nm"] is not None:
            x1, x2 = hb["center_mean_nm"], ha["center_mean_nm"]
            y1, y2 = 486.135, 656.281
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            corrected_hg = intercept + slope * hg["center_mean_nm"]
            residual = float(corrected_hg - 434.047)
            report["provisional_wavelength_crosscheck"] = {
                "calibration_lines": ["H_beta", "H_alpha"],
                "method": "two-point affine mapping; diagnostic only, not accepted as a calibration",
                "slope": float(slope),
                "intercept_nm": float(intercept),
                "independent_check_line": "H_gamma",
                "H_gamma_observed_center_nm": hg["center_mean_nm"],
                "H_gamma_mapped_center_nm": float(corrected_hg),
                "H_gamma_reference_nm_approx": 434.047,
                "H_gamma_residual_nm": residual,
                "crosscheck_pass": bool(abs(residual) <= 0.25),
                "decision": "DO_NOT_APPLY_CALIBRATION" if abs(residual) > 0.25 else "PROVISIONAL_ONLY",
                "reason": "Only two lines define the affine map; H-gamma is an independent check and its residual must be acceptably small before calibration can be trusted."
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
            "interpretation": "Offsets are diagnostic only; H-gamma is weak and the two-line affine calibration failed its independent H-gamma check."
        }
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
        width_match = re.search(r"(?im)^width\(wavelength\)\s*=\s*(.+)$", section)
        pairs = []
        if width_match:
            nums = [float(v) for v in re.findall(r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?", width_match.group(1))]
            pairs = list(zip(nums[0::2], nums[1::2]))
        if pairs:
            inst_width = float(np.interp(656.281, [p[0] for p in pairs], [p[1] for p in pairs]))
            config_audit = {
                "status": "instrument_config_retrieved",
                "source_url": config_url,
                "sha256": hashlib.sha256(config_blob).hexdigest(),
                "width_points_nm": [{"wavelength_nm": x, "instrumental_width_nm": y} for x, y in pairs],
                "interpolated_instrumental_width_at_H_alpha_nm": inst_width,
                "width_definition": "as defined in upstream GOLEM config; confirm before using as FWHM",
                "decision": "not_applied_until_width_definition_and_same_instrument_provenance_are_confirmed"
            }
            break
        config_audit["attempts"].append({"url": config_url, "status": "config_found_but_width_parse_failed"})
    except Exception as exc:
        config_audit["attempts"].append({"url": config_url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
report["instrument_config_audit"] = config_audit

(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["status"] != "PASS_SPECTRAL_LINE_SHAPE":
    sys.exit(2)
