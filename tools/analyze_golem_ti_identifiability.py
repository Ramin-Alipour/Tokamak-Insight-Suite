#!/usr/bin/env python3
"""Audit whether GOLEM shot 30622 Balmer widths can identify ion temperature.

The script fits H-alpha, H-beta and H-gamma independently, calculates the
Doppler-only apparent temperatures as a diagnostic, and tests line-width
consistency and spectral sampling. Apparent values are explicitly not accepted
as physical Ti unless sampling and instrumental-response requirements are met.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.optimize import curve_fit

SHOT = 30622
URL = (
    f"https://golem.fjfi.cvut.cz/operation/shots/{SHOT}/diagnostics/"
    "Radiation/1111Spectrometer.ON/data/spectra.txt_Data_.txt"
)
EXPECTED_SHA256 = "3450043b07322fc5516462cc826e53ce6e8c6bf447f61fc63b999a0755d3bc3d"
OUT = Path("artifacts/golem-ti-identifiability")
OUT.mkdir(parents=True, exist_ok=True)

def get_raw() -> bytes:
    req = Request(URL, headers={"User-Agent": "Tokamak-Insight-Suite-Ti-identifiability/1.0"})
    with urlopen(req, timeout=30) as response:
        return response.read()

def parse_spectrum(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    rows = []
    timestamps = []
    for line in text.splitlines():
        if line.startswith("Time stamps [ms]:"):
            payload = line.split(":", 1)[1].strip().strip("[]")
            try:
                timestamps = [float(v) for v in payload.split()]
            except ValueError:
                timestamps = []
            continue
        try:
            vals = [float(token) for token in line.split()]
        except ValueError:
            continue
        if len(vals) == 30 and np.all(np.isfinite(vals)):
            rows.append(vals)
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 100 or arr.shape[1] != 30:
        raise ValueError(f"Unexpected GOLEM spectrum matrix shape {arr.shape}")
    return arr[:, 0], arr[:, 1:], timestamps

def gaussian_linear(x, amp, center, sigma, baseline, slope):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline + slope * (x - center)

def fit_line(wavelength, intensity, name, reference_nm, window):
    mask = (wavelength >= window[0]) & (wavelength <= window[1])
    x, y = wavelength[mask], intensity[mask]
    if len(x) < 8:
        raise ValueError(f"Too few samples for {name}: {len(x)}")
    baseline = float(np.percentile(y, 20))
    p0 = [max(float(np.max(y) - baseline), 1.0), float(x[np.argmax(y)]), 0.7, baseline, 0.0]
    params, covariance = curve_fit(
        gaussian_linear, x, y, p0=p0, maxfev=50000,
        bounds=([0, window[0], 0.03, -1e5, -1e4],
                [1e7, window[1], 8.0, 1e5, 1e4]),
    )
    amp, center, sigma, baseline, slope = map(float, params)
    fwhm = 2.354820045 * sigma
    fitted = gaussian_linear(x, *params)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    # For a hydrogen atom, kT is in eV and m_H*c^2 in eV.
    mHc2_eV = 1.00784 * 931.49410242e6
    apparent_Ti = (fwhm / reference_nm) ** 2 * mHc2_eV / (8.0 * np.log(2.0))
    return {
        "reference_air_wavelength_nm": reference_nm,
        "fit_amplitude": amp,
        "fit_baseline": baseline,
        "fit_sigma_nm": sigma,
        "fitted_center_nm": center,
        "center_offset_nm": center - reference_nm,
        "observed_gaussian_fwhm_nm": fwhm,
        "fwhm_in_sample_intervals": fwhm / float(np.median(np.diff(wavelength))),
        "doppler_only_apparent_Ti_eV_not_accepted": apparent_Ti,
        "fit_r_squared": r2,
        "covariance_finite": bool(np.all(np.isfinite(covariance))),
        "fit_point_count": int(len(x)),
    }

raw = get_raw()
digest = hashlib.sha256(raw).hexdigest()
if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
    raise ValueError(f"GOLEM raw spectrum SHA-256 mismatch: {digest}")
wavelength, spectra, timestamps = parse_spectrum(raw)
np.savetxt(OUT / "golem_30622_spectrum_matrix.csv",
           np.column_stack([wavelength, spectra]), delimiter=",",
           header="wavelength_nm," + ",".join(f"exposure_{i+1}" for i in range(spectra.shape[1])),
           comments="")
mean_spectrum = np.mean(spectra, axis=1)
step = float(np.median(np.diff(wavelength)))
lines = {
    "H_alpha": (656.281, (650.0, 662.0)),
    "H_beta": (486.134, (480.0, 492.0)),
    "H_gamma": (434.047, (429.0, 439.0)),
}
line_results = {
    name: fit_line(wavelength, mean_spectrum, name, ref, window)
    for name, (ref, window) in lines.items()
}
# Heavy-ion lines have much smaller Doppler widths than hydrogen at the same Ti.
# Their measured widths can be used as a rough plasma-line proxy for instrumental
# broadening, but not as an independently measured lamp LSF.
proxy_lines = {
    "OII_4417": (441.7, (438.0, 445.0)),
    "NII_4899": (489.9, (487.0, 493.0)),
    "NII_5002": (500.2, (497.0, 503.0)),
    "OII_5392": (539.2, (536.0, 543.0)),
}
proxy_results = {}
for name, (ref, window) in proxy_lines.items():
    try:
        proxy_results[name] = fit_line(wavelength, mean_spectrum, name, ref, window)
    except Exception as exc:
        proxy_results[name] = {"fit_error": f"{type(exc).__name__}: {exc}"}
apparent = [v["doppler_only_apparent_Ti_eV_not_accepted"] for v in line_results.values()]
min_app, max_app = min(apparent), max(apparent)
widths = [v["observed_gaussian_fwhm_nm"] for v in line_results.values()]

# Independent leave-one-line-out affine wavelength-calibration check:
# use two Balmer centroids to calibrate, then predict the third reference line.
line_names = list(lines)
measured_centers = np.array([line_results[name]["fitted_center_nm"] for name in line_names])
reference_centers = np.array([line_results[name]["reference_air_wavelength_nm"] for name in line_names])
loo_calibration = []
for held_out in range(len(line_names)):
    training = np.array([i for i in range(len(line_names)) if i != held_out])
    slope, intercept = np.polyfit(measured_centers[training], reference_centers[training], 1)
    predicted = float(slope * measured_centers[held_out] + intercept)
    error = predicted - float(reference_centers[held_out])
    loo_calibration.append({
        "held_out_line": line_names[held_out],
        "training_lines": [line_names[i] for i in training],
        "affine_slope": float(slope),
        "affine_intercept_nm": float(intercept),
        "predicted_reference_nm": predicted,
        "prediction_error_nm": error,
        "prediction_error_in_sampling_intervals": error / step,
        "within_one_sampling_interval": abs(error) <= step,
    })
all_slope, all_intercept = np.polyfit(measured_centers, reference_centers, 1)
all_calibration_residuals = (all_slope * measured_centers + all_intercept) - reference_centers
report = {
    "analysis": "Balmer multi-line ion-temperature identifiability audit",
    "shot": SHOT,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_url": URL,
    "source_sha256": digest,
    "source_bytes": len(raw),
    "wavelength_samples": int(wavelength.size),
    "exposure_count": int(spectra.shape[1]),
    "timestamp_count": int(len(timestamps)),
    "wavelength_range_nm": [float(wavelength.min()), float(wavelength.max())],
    "median_sampling_interval_nm": step,
    "line_fits": line_results,
    "heavy_ion_line_width_proxies_not_independent_LSF": proxy_results,
    "wavelength_calibration_cross_check": {
        "method": "leave-one-Balmer-line-out affine calibration; two lines fit, third predicted",
        "line_centroids_used_nm": {name: float(line_results[name]["fitted_center_nm"]) for name in line_names},
        "reference_wavelengths_nm": {name: float(line_results[name]["reference_air_wavelength_nm"]) for name in line_names},
        "leave_one_out_predictions": loo_calibration,
        "all_line_fit_slope": float(all_slope),
        "all_line_fit_intercept_nm": float(all_intercept),
        "all_line_residuals_nm": {name: float(all_calibration_residuals[i]) for i, name in enumerate(line_names)},
        "calibration_generalizes_within_one_sampling_interval": all(item["within_one_sampling_interval"] for item in loo_calibration),
        "limitation": "Only three broad, sparsely sampled plasma lines are available; this is a stress test, not a substitute for a lamp-standard wavelength calibration."
    },
    "multi_line_doppler_consistency": {
        "apparent_Ti_min_eV_not_accepted": min_app,
        "apparent_Ti_max_eV_not_accepted": max_app,
        "max_to_min_ratio": max_app / min_app if min_app > 0 else None,
        "observed_widths_scale_with_wavelength_as_expected_for_pure_Doppler_broadening": (
            max(widths) / min(widths) < 1.25
        ),
    },
    "instrument_width_sensitivity_H_alpha": [],
    "acceptance": {
        "ti_identifiable_from_this_dataset": False,
        "reason": [
            "All fitted Balmer widths span fewer than four wavelength-sampling intervals.",
            "The fitted H-alpha/H-beta/H-gamma widths do not provide a mutually consistent pure-Doppler temperature.",
            "No independently measured instrument line-spread function (LSF) for this exact setup/shot is supplied.",
            "The 20.8 eV GOLEM shot-page scalar is electron temperature, not an ion-temperature reference."
        ],
        "required_next_input": "A higher-resolution Balmer spectrum plus an independently measured LSF for the same spectrometer configuration, or an independently documented Ti reference for the same shot and time window."
    },
}
h_alpha = line_results["H_alpha"]
obs = h_alpha["observed_gaussian_fwhm_nm"]
lam = h_alpha["reference_air_wavelength_nm"]
mHc2_eV = 1.00784 * 931.49410242e6
for inst_fwhm in [0.0, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.18, 1.19, 1.2]:
    rem2 = obs * obs - inst_fwhm * inst_fwhm
    t = (rem2 / (lam * lam)) * mHc2_eV / (8.0 * np.log(2.0)) if rem2 > 0 else None
    report["instrument_width_sensitivity_H_alpha"].append({
        "assumed_instrument_FWHM_nm_not_measured": inst_fwhm,
        "resulting_gaussian_quadrature_Ti_eV_not_accepted": t,
        "note": "Sensitivity illustration only; assumes Gaussian broadening and omits Stark/other broadening."
    })
proxy_widths = [
    item["observed_gaussian_fwhm_nm"]
    for item in proxy_results.values()
    if "observed_gaussian_fwhm_nm" in item
    and item.get("fit_r_squared") is not None
    and item["fit_r_squared"] >= 0.8
]
if proxy_widths:
    proxy_lsf = float(np.median(proxy_widths))
    rem2_proxy = obs * obs - proxy_lsf * proxy_lsf
    report["plasma_line_width_proxy_summary"] = {
        "n_lines_used": len(proxy_widths),
        "line_fwhm_nm": proxy_widths,
        "median_proxy_FWHM_nm": proxy_lsf,
        "median_proxy_FWHM_in_sampling_intervals": proxy_lsf / step,
        "Ti_from_H_alpha_after_proxy_quadrature_eV_not_accepted": (
            float((rem2_proxy / (lam * lam)) * mHc2_eV / (8.0 * np.log(2.0))
                  if rem2_proxy > 0 else 0.0)
        ),
        "limitation": "These are plasma emission lines, not a lamp-standard LSF; blending, Stark broadening, and low sampling may bias the proxy."
    }
else:
    report["plasma_line_width_proxy_summary"] = {
        "n_lines_used": 0,
        "limitation": "No heavy-ion line fit passed the basic fit-quality threshold."
    }
report["instrument_width_required_for_target_Ti_H_alpha"] = []
for target_ti in [20.8, 50.0, 100.0, 300.0]:
    doppler_fwhm = lam * np.sqrt(8.0 * np.log(2.0) * target_ti / mHc2_eV)
    required_lsf = float(np.sqrt(max(obs * obs - doppler_fwhm * doppler_fwhm, 0.0)))
    report["instrument_width_required_for_target_Ti_H_alpha"].append({
        "target_Ti_eV": target_ti,
        "expected_Doppler_FWHM_nm": float(doppler_fwhm),
        "required_instrument_FWHM_nm": required_lsf,
        "interpretation": "Required Gaussian LSF if the entire excess width is Doppler broadening; sensitivity calculation, not a measured calibration."
    })
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
