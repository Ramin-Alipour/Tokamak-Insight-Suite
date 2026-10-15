from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tis.probes import fit_swept_ball_pen_ion_temperature


def _bpp_model(v, ti=21.0, phi=7.15, isat_plus=3.08, slope=0.003, alpha=0.6):
    return np.exp(alpha) * isat_plus * (1.0 + slope * (v - phi)) - isat_plus * np.exp((phi - v) / ti)


def test_swept_bpp_fit_recovers_known_synthetic_temperature():
    voltage = np.linspace(10.0, 85.0, 40)
    current = _bpp_model(voltage)
    result = fit_swept_ball_pen_ion_temperature(
        voltage, current, alpha_bpp=0.6, fit_min_voltage_V=10.0,
        initial_ti_eV=18.0, plasma_potential_guess_V=7.15,
    )
    assert abs(result["Ti_eV"] - 21.0) < 0.1
    assert result["covariance_finite"]
    assert result["n_points_used"] == 40


def test_swept_bpp_fit_is_consistent_with_digitized_published_curve():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "validation" / "golem_bpp_31992_figure3_digitized.csv"
    with fixture.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(line for line in f if not line.startswith("#"))]
    result = fit_swept_ball_pen_ion_temperature(
        [float(r["voltage_V"]) for r in rows],
        [float(r["current_mA"]) for r in rows],
        alpha_bpp=0.6, fit_min_voltage_V=10.0,
        initial_ti_eV=21.0, plasma_potential_guess_V=7.15,
    )
    # The source is a plot, not raw samples, so this deliberately uses a broad
    # tolerance and must not be presented as raw-data reproduction.
    assert 15.0 <= result["Ti_eV"] <= 30.0
    assert abs(result["Ti_eV"] - 21.0) <= 10.0
