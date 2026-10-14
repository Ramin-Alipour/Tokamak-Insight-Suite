#!/usr/bin/env python3
"""Reproduce published EAST probe fluctuation statistics from independent raw data.

Dataset: Pécseli et al. et al., Zenodo DOI 10.5281/zenodo.18473625.
Paper: "Small scale turbulence and plasma transport in the EAST tokamak",
Plasma Physics and Controlled Fusion (2026), DOI 10.1088/1361-6587/ae4f20.
Uses the published index window and fourth-order polynomial detrending.
"""
from __future__ import annotations
import io, json, math, re, urllib.request, zipfile
from pathlib import Path
from urllib.parse import quote
import numpy as np

from tis.core import electric_field_from_potential_difference, signal_quality

BASE = "https://zenodo.org/records/18473625/files/"
FILES = ["22010ne.txt", "22010Te.txt", "22010vf1.txt", "22010vf2.txt", "22010vf3.txt"]
OUT = Path("artifacts/east-probe-validation")
OUT.mkdir(parents=True, exist_ok=True)
START, STOP = 275000, 309600  # stop is exclusive; 34,600 samples
DTREND_ORDER = 4
EXPECTED = {
    "ne": {"std": 3.9e17, "unit": "m^-3", "tolerance_fraction": 0.10, "paper": "sigma_ne = 39 x 10^16 m^-3"},
    "Te": {"std": 9.6, "unit": "eV", "tolerance_fraction": 0.05, "paper": "sigma_Te = 9.6 eV"},
    "vf1": {"std": 8.7, "unit": "V", "tolerance_fraction": 0.05, "paper": "sigma_vf1 = 8.7 V"},
    "vf2": {"std": 7.9, "unit": "V", "tolerance_fraction": 0.05, "paper": "sigma_vf2 = 7.9 V"},
    "vf3": {"std": 6.8, "unit": "V", "tolerance_fraction": 0.05, "paper": "sigma_vf3 = 6.8 V"},
}
FIELD_EXPECTED = {"E12": 1.1e3, "E13": 0.6e3, "E23": 1.0e3}

def fetch(name: str) -> bytes:
    url = BASE + quote(name, safe="") + "?download=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Tokamak-Insight-Suite-validation/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()

def detrend4(x: np.ndarray) -> np.ndarray:
    # Normalized coordinate improves conditioning; matches fourth-order polynomial removal.
    t = np.linspace(-1.0, 1.0, len(x))
    coeff = np.polynomial.polynomial.polyfit(t, x, DTREND_ORDER)
    return x - np.polynomial.polynomial.polyval(t, coeff)

results, cleaned = [], {}
for filename in FILES:
    raw = fetch(filename)
    (OUT / filename).write_bytes(raw)
    values = np.fromstring(raw.decode("utf-8-sig"), sep=" ")
    if values.size < STOP:
        raise RuntimeError(f"{filename}: {values.size} rows, expected at least {STOP}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{filename}: non-finite values found")
    key = filename.removeprefix("22010").removesuffix(".txt")
    segment = values[START:STOP]
    detrended = detrend4(segment)
    cleaned[key] = detrended
    # The density series is stored in units of 10^19 m^-3; the other series
    # are in eV (Te) and V (floating potentials), as cross-checked against paper.
    scale = 1e19 if key == "ne" else 1.0
    # Exercise the actual TIS implementation, then cross-check it independently
    # against NumPy's sample standard deviation before comparing to the paper.
    tis_quality = signal_quality(detrended)
    observed = float(tis_quality["std_sample"] * scale)
    independent_std = float(np.std(detrended, ddof=1) * scale)
    software_crosscheck_error = abs(observed - independent_std) / max(abs(independent_std), np.finfo(float).tiny)
    if not math.isclose(observed, independent_std, rel_tol=1e-12, abs_tol=1e-14):
        raise RuntimeError(f"{key}: TIS signal_quality disagrees with independent sample standard deviation")
    reference = EXPECTED[key]["std"]
    rel_error = abs(observed - reference) / abs(reference)
    limit = EXPECTED[key]["tolerance_fraction"]
    status = "PASS" if rel_error <= limit else "FAIL"
    item = {
        "file": filename, "samples_total": int(values.size), "window_indices": [START, STOP],
        "window_samples": int(segment.size), "detrending": f"polynomial order {DTREND_ORDER}",
        "raw_mean_source_units": float(np.mean(segment)),
        "raw_std_source_units": float(np.std(segment, ddof=1)),
        "detrended_std": observed, "unit": EXPECTED[key]["unit"],
        "tis_core_function": "tis.core.signal_quality",
        "independent_numpy_sample_std": independent_std,
        "software_vs_independent_relative_error": software_crosscheck_error,
        "published_reference": reference, "published_statement": EXPECTED[key]["paper"],
        "relative_error": rel_error, "acceptance_limit_fraction": limit, "comparison": status,
    }
    if key == "ne":
        relative_fluctuation = float(tis_quality["std_sample"] / np.mean(segment))
        relative_reference = 0.15
        relative_tolerance = 0.20  # paper reports an approximate ~15% relative fluctuation
        relative_error = abs(relative_fluctuation - relative_reference) / relative_reference
        relative_status = "PASS" if relative_error <= relative_tolerance else "FAIL"
        item["relative_density_fluctuation"] = relative_fluctuation
        item["published_relative_density_fluctuation_approx"] = relative_reference
        item["relative_fluctuation_relative_error"] = relative_error
        item["relative_fluctuation_acceptance_limit_fraction"] = relative_tolerance
        item["relative_fluctuation_comparison"] = relative_status
        print(f"{relative_status} relative density fluctuation: {relative_fluctuation:.6g}; "
              f"published≈{relative_reference:.6g}; relative error={relative_error:.2%}")
    results.append(item)
    print(f"{status} {key}: reproduced sigma={observed:.6g} {EXPECTED[key]['unit']}; "
          f"published={reference:.6g}; relative error={rel_error:.2%}")

# Reproduce electric-field fluctuation levels from detrended floating potentials.
field_defs = {
    "E12": ("vf1", "vf2", 0.008),
    "E13": ("vf1", "vf3", 0.016),
    "E23": ("vf2", "vf3", 0.008),
}
fields = []
for key, (channel_a, channel_b, separation_m) in field_defs.items():
    # The TIS electric-field routine is the subject under test. The paper
    # compares fluctuation magnitudes, so field_sign=+1 matches the explicit
    # difference convention here; the sign does not affect standard deviation.
    tis_field = electric_field_from_potential_difference(
        cleaned[channel_a], cleaned[channel_b], separation_m, field_sign=1
    )
    independent_field = (cleaned[channel_b] - cleaned[channel_a]) / separation_m
    observed = float(signal_quality(tis_field)["std_sample"])
    independent_std = float(np.std(independent_field, ddof=1))
    software_crosscheck_error = abs(observed - independent_std) / max(abs(independent_std), np.finfo(float).tiny)
    if not math.isclose(observed, independent_std, rel_tol=1e-12, abs_tol=1e-10):
        raise RuntimeError(f"{key}: TIS electric-field result disagrees with independent difference calculation")
    reference = FIELD_EXPECTED[key]
    rel_error = abs(observed - reference) / reference
    # Article reports approximate values at two significant digits.
    limit = 0.15
    status = "PASS" if rel_error <= limit else "FAIL"
    fields.append({"quantity": key, "observed_std_V_per_m": observed,
                   "tis_core_functions": ["tis.core.electric_field_from_potential_difference", "tis.core.signal_quality"],
                   "independent_numpy_std_V_per_m": independent_std,
                   "software_vs_independent_relative_error": software_crosscheck_error,
                   "published_reference_V_per_m": reference, "relative_error": rel_error,
                   "acceptance_limit_fraction": limit, "comparison": status})
    print(f"{status} {key}: sigma_E={observed:.6g} V/m; published≈{reference:.6g}; error={rel_error:.2%}")

readme_name = "readme - EAST tokamak plasma probe data.docx"
readme_bytes = fetch(readme_name)
(OUT / "dataset_readme.docx").write_bytes(readme_bytes)
with zipfile.ZipFile(io.BytesIO(readme_bytes)) as zf:
    xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
text = re.sub(r"</w:p>", "\\n", xml)
text = re.sub(r"<[^>]+>", "", text)
text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&apos;", "'"))
(OUT / "dataset_readme_extracted.txt").write_text(text, encoding="utf-8")

passed = sum(x["comparison"] == "PASS" for x in results) + sum(x["comparison"] == "PASS" for x in fields)
passed += sum(x.get("relative_fluctuation_comparison") == "PASS" for x in results)
total = len(results) + len(fields) + sum("relative_fluctuation_comparison" in x for x in results)
report = {
    "dataset": "EAST tokamak plasma probe data", "doi": "10.5281/zenodo.18473625",
    "paper_doi": "10.1088/1361-6587/ae4f20", "window_indices_python": [START, STOP],
    "window_duration_ms_at_2MHz": [137.5, 154.8], "detrending": "fourth-order polynomial",
    "density_source_scale": "22010ne.txt values multiplied by 1e19 m^-3",
    "results": results, "electric_field_results": fields,
    "passed_comparisons": passed, "total_comparisons": total,
    "status": "PASS" if passed == total else "FAIL",
    "interpretation": "The TIS signal_quality and electric_field_from_potential_difference functions were run on the independent published EAST probe dataset and cross-checked against separately expressed NumPy calculations before comparison to paper values. This validates only these tested routines and settings, not every TIS module or diagnostic calibration.",
}
(OUT / "east_probe_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
if passed != total:
    raise SystemExit(f"Scientific benchmark failed: {passed}/{total} comparisons passed")
print(f"SUMMARY: {passed}/{total} external scientific comparisons passed.")
