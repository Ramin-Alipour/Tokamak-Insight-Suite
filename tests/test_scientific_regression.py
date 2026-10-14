"""Regression tests for scientific invariants and defensive input validation.

These are software-level checks using analytical/synthetic values, not validation
against a particular tokamak or diagnostic calibration.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tis.core import (
    AMU, C_LIGHT, E_CHARGE, EPS0, M_E, MU0,
    corrected_line_ratio, derive_parameters, doppler_temperature_eV,
    doppler_velocity_m_s, electron_cyclotron_resonance_field,
    exponential_heat_flux_fit, floating_potential, integrate_profile,
    linear_phase_density, neutron_rate_from_counts, reflectometry_cutoff_density,
    spectrum_moments, xray_spectrum_summary,
)
from tis.catalog import DIAGNOSTICS, FORMULAS


def test_doppler_temperature_matches_thermal_gaussian_sigma_relation():
    sigma_lambda, lambda0, mass_amu = 0.01, 500.0, 40.0
    expected = mass_amu * AMU * C_LIGHT**2 / E_CHARGE * (sigma_lambda / lambda0)**2
    assert doppler_temperature_eV(sigma_lambda, lambda0, mass_amu) == pytest.approx(expected)


def test_doppler_temperature_zero_width_is_zero():
    assert doppler_temperature_eV(0.0, 500.0, 40.0) == 0.0


@pytest.mark.parametrize("args", [(-1, 500, 40), (1, 0, 40), (1, 500, 0)])
def test_doppler_temperature_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        doppler_temperature_eV(*args)


def test_doppler_velocity_uses_small_shift_approximation():
    assert doppler_velocity_m_s(0.1, 500.0) == pytest.approx(C_LIGHT * 0.1 / 500.0)


def test_derived_plasma_pressure_uses_electron_volts_as_energy_per_charge():
    ne, te = 2.0e19, 50.0
    result = derive_parameters(ne, te, 0, 1e5, 1, 1.5, 0.5, 2.0)
    assert result["electron_pressure_Pa"] == pytest.approx(ne * te * E_CHARGE)
    assert result["debye_length_m"] == pytest.approx(np.sqrt(EPS0 * te / (ne * E_CHARGE)))


def test_energy_confinement_time_requires_positive_loss_power():
    assert derive_parameters(1e19, 10, 10, 1e5, 1, 1, .3, 2, 1000, 500)["energy_confinement_time_s"] == 2
    assert derive_parameters(1e19, 10, 10, 1e5, 1, 1, .3, 2, 1000, 0)["energy_confinement_time_s"] is None


def test_invalid_plasma_radius_is_rejected():
    with pytest.raises(ValueError):
        derive_parameters(1e19, 10, 10, 1e5, 1, 0, .3, 2)


def test_floating_potential_interpolates_zero_crossing():
    assert floating_potential([-2, -1, 0, 1, 2], [-2, -1, 1, 2, 3]) == pytest.approx(-0.5)


def test_profile_integral_is_coordinate_aware():
    result = integrate_profile([0, 1, 2], [0, 1, 2])
    assert result["integral_trapezoid"] == pytest.approx(2.0)


def test_profile_integral_rejects_duplicate_coordinates():
    with pytest.raises(ValueError):
        integrate_profile([0, 1, 1], [1, 2, 3])


def test_reflectometry_cutoff_density_scales_with_frequency_squared():
    a = reflectometry_cutoff_density(60e9, mode="O")
    b = reflectometry_cutoff_density(120e9, mode="O")
    assert b["cutoff_density_m-3"] == pytest.approx(4 * a["cutoff_density_m-3"], rel=1e-10)


def test_electron_cyclotron_resonance_field_scales_with_frequency():
    a = electron_cyclotron_resonance_field(60e9, harmonic=1)
    b = electron_cyclotron_resonance_field(120e9, harmonic=1)
    assert b["resonance_field_T"] == pytest.approx(2 * a["resonance_field_T"], rel=1e-10)


def test_neutron_count_rate_efficiency_correction():
    r = neutron_rate_from_counts(1000, efficiency=0.2, geometric_factor=0.5, calibration_factor=2)
    assert r["estimated_source_rate_s-1"] == pytest.approx(20000)


def test_line_ratio_response_correction():
    r = corrected_line_ratio(20, 10, response_1=2, response_2=1)
    assert r["instrument_response_corrected_ratio"] == pytest.approx(1)


def test_spectral_moment_centroid_for_symmetric_peak():
    r = spectrum_moments([-2, -1, 0, 1, 2], [0, 1, 2, 1, 0])
    assert r["mean_energy_eV"] == pytest.approx(0.0)


def test_xray_spectrum_summary_reports_positive_total_counts():
    r = xray_spectrum_summary([1, 2, 3], [2, 5, 2])
    assert r["integrated_counts_keV"] == pytest.approx(7.0)


def test_catalog_and_formula_registry_are_nonempty_and_structured():
    assert len(DIAGNOSTICS) >= 80
    assert len(FORMULAS) >= 35
    assert all(d.get("diagnostic") and d.get("family") and d.get("caveat") for d in DIAGNOSTICS)
    assert all(isinstance(f, dict) and f for f in FORMULAS)


def test_registry_json_files_are_valid_json():
    root = Path(__file__).parents[1]
    for name in ("DIAGNOSTIC_REGISTRY.json", "FORMULA_REGISTRY.json"):
        with (root / name).open(encoding="utf-8") as stream:
            parsed = json.load(stream)
        assert isinstance(parsed, (list, dict))
        assert parsed
