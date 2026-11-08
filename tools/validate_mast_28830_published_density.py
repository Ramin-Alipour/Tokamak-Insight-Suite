#!/usr/bin/env python3
"""Provisional numerical cross-check of public MAST 28830 data against a paper.

Reference: Walkden et al., arXiv:1411.7298, Table II.
The paper does not state the exact time averaging interval for the table, so this
is explicitly a rounded-value consistency check, not a full raw-probe validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("artifacts/mast-bpp-discovery")
WINDOW_S = (0.15, 0.25)
REFERENCE = {
    "edge_density_m3": 0.8e19,
    "line_integrated_density_m2": 110e18,
}
TOLERANCE_RELATIVE = 0.10


def relative_difference(value: float, reference: float) -> float:
    return float((value - reference) / reference)


def main() -> None:
    profile_path = BASE / "mast_28830_thomson_profiles.csv"
    interferometer_path = BASE / "mast_28830_interferometer_density.csv"
    if not profile_path.exists() or not interferometer_path.exists():
        raise FileNotFoundError("Required shot-28830 Thomson/interferometer exports are missing.")

    profile = pd.read_csv(profile_path)
    interferometer = pd.read_csv(interferometer_path)
    lo, hi = WINDOW_S
    profile = profile[(profile["time_s"] >= lo) & (profile["time_s"] < hi)].copy()
    interferometer = interferometer[
        (interferometer["time_s"] >= lo) & (interferometer["time_s"] < hi)
    ].copy()
    if profile.empty or interferometer.empty:
        raise ValueError(f"No measurements in requested time interval {lo}-{hi} s.")

    # One Thomson measurement per time slice: select the closest available radial
    # channel to the published R=1.38 m location, rather than averaging channels.
    nearest_rows = []
    for time_s, group in profile.groupby("time_s", sort=True):
        finite = group[
            np.isfinite(group["radius_m"])
            & np.isfinite(group["ne_m3"])
            & np.isfinite(group["te_eV"])
        ]
        if finite.empty:
            continue
        idx = (finite["radius_m"] - 1.38).abs().idxmin()
        row = finite.loc[idx]
        if abs(float(row["radius_m"]) - 1.38) <= 0.01:
            nearest_rows.append({
                "time_s": float(time_s),
                "radius_m": float(row["radius_m"]),
                "te_eV": float(row["te_eV"]),
                "ne_m3": float(row["ne_m3"]),
                "ne_error_m3": float(row["ne_error"]) if "ne_error" in row and np.isfinite(row["ne_error"]) else float("nan"),
            })
    if len(nearest_rows) < 5:
        raise ValueError("Fewer than five finite Thomson samples within 1 cm of R=1.38 m.")
    near_edge = pd.DataFrame(nearest_rows)
    measured_edge_density = float(near_edge["ne_m3"].median())
    measured_edge_te = float(near_edge["te_eV"].median())
    edge_measurement_error = float(near_edge["ne_error_m3"].median())
    edge_temporal_mad = float(np.median(np.abs(near_edge["ne_m3"] - measured_edge_density)))
    edge_temporal_sigma = 1.4826 * edge_temporal_mad
    measured_line_density = float(
        interferometer.loc[
            np.isfinite(interferometer["line_density_raw"]), "line_density_raw"
        ].median()
    )
    line_values = interferometer.loc[
        np.isfinite(interferometer["line_density_raw"]), "line_density_raw"
    ].to_numpy(dtype=float)
    line_temporal_mad = float(np.median(np.abs(line_values - measured_line_density)))
    line_temporal_sigma = 1.4826 * line_temporal_mad
    edge_rel = relative_difference(measured_edge_density, REFERENCE["edge_density_m3"])
    line_rel = relative_difference(
        measured_line_density, REFERENCE["line_integrated_density_m2"]
    )

    result = {
        "benchmark": "MAST shot 28830: Thomson edge density and interferometer line density",
        "publication": {
            "title": "Profile measurements in the plasma edge of MAST using a ball pen probe",
            "url": "https://arxiv.org/abs/1411.7298",
            "reference_location": "Table II; edge density at R=1.38 m and line-integrated density",
            "published_values": REFERENCE,
        },
        "source": {
            "dataset": "UKAEA FAIR-MAST Level-1",
            "shot": 28830,
            "shot_url": "https://s3.echo.stfc.ac.uk/mast/level1/shots/28830.zarr",
            "groups": ["ayc", "ane"],
            "signals": ["ayc.te", "ayc.ne", "ayc.radius", "ane.density"],
            "classification": "archived Thomson-derived profiles plus independent interferometer line-density signal",
        },
        "selection": {
            "time_window_s": [lo, hi],
            "radial_reference_m": 1.38,
            "median_radius_m": float(near_edge["radius_m"].median()),
            "n_thomson_time_slices": int(near_edge["time_s"].nunique()),
            "n_interferometer_samples": int(interferometer["line_density_raw"].count()),
            "important_limit": "The publication does not state the exact time averaging interval for Table II; this windowed comparison is provisional.",
        },
        "measured": {
            "edge_density_m3": measured_edge_density,
            "edge_temperature_eV_median": measured_edge_te,
            "line_integrated_density_m2": measured_line_density,
        },
        "uncertainty_and_variability": {
            "edge_density_median_reported_ne_error_m3": edge_measurement_error,
            "edge_density_temporal_robust_sigma_m3": edge_temporal_sigma,
            "edge_density_discrepancy_over_reported_error": (measured_edge_density - REFERENCE["edge_density_m3"]) / edge_measurement_error,
            "line_density_temporal_robust_sigma_m2": line_temporal_sigma,
            "line_density_discrepancy_over_temporal_robust_sigma": (measured_line_density - REFERENCE["line_integrated_density_m2"]) / line_temporal_sigma,
            "note": "The edge-density uncertainty is the archive's ne_error at the selected radial channel. The line-density spread is temporal variability, not a separately documented instrument calibration uncertainty.",
        },
        "comparison": {
            "edge_density_relative_difference": edge_rel,
            "edge_density_absolute_relative_difference": abs(edge_rel),
            "line_density_relative_difference": line_rel,
            "line_density_absolute_relative_difference": abs(line_rel),
            "tolerance_relative": TOLERANCE_RELATIVE,
        },
        "classification": "provisional_cross_diagnostic_consistency_check_not_full_raw_probe_validation",
        "full_validation_blocker": "The FAIR-MAST Level-1 export exposes reciprocating-probe position but not the paper's raw BPP and LP voltage traces; Eq. (3) cannot be independently reconstructed from these files.",
    }
    result["within_tolerance"] = (
        abs(edge_rel) <= TOLERANCE_RELATIVE and abs(line_rel) <= TOLERANCE_RELATIVE
    )
    output = BASE / "mast_28830_published_density_crosscheck.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["within_tolerance"]:
        raise AssertionError(
            "MAST 28830 archived density measurements differ from the paper's rounded table values by more than 10%."
        )


if __name__ == "__main__":
    main()
