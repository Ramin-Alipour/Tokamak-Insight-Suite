"""Regression tests for physically singular uncertainty and estimator edge cases."""
import math

import numpy as np
import pytest

from tokamak_analysis.formula_library import (
    bolometry_regularized_inverse,
    faraday_induction,
    magnitude_squared_coherence,
    magnetic_energy_density,
    ion_saturation_particle_flux_proxy,
)


def test_magnetic_energy_density_first_order_uncertainty_reference():
    field = 2.0
    sigma = 0.01
    result = magnetic_energy_density(field, sigma_magnetic_field_T=sigma)
    assert result["magnetic_energy_density_J_m3"] > 0.0
    assert result["uncertainty_magnetic_energy_density_J_m3"] == pytest.approx(
        abs(field) * sigma / 1.25663706212e-6, rel=1e-9
    )


def test_magnetic_energy_density_does_not_report_false_zero_uncertainty_at_zero_field():
    assert magnetic_energy_density(0.0)["magnetic_energy_density_J_m3"] == 0.0
    with pytest.raises(ValueError, match="undefined at B=0"):
        magnetic_energy_density(0.0, sigma_magnetic_field_T=0.1)


def test_particle_flux_proxy_rejects_singular_linearized_uncertainty_at_zero_current():
    assert ion_saturation_particle_flux_proxy(0.0, 1e-4)["ion_saturation_particle_flux_m2_s"] == 0.0
    with pytest.raises(ValueError, match="undefined at zero"):
        ion_saturation_particle_flux_proxy(
            0.0, 1e-4, sigma_saturation_current_A=1e-3, sigma_effective_area_m2=1e-6
        )


def test_coherence_rejects_constant_signal_and_returns_bounded_finite_values():
    t = np.arange(128, dtype=float)
    with pytest.raises(ValueError, match="constant"):
        magnitude_squared_coherence(np.ones(128), np.sin(t), 1000.0)
    result = magnitude_squared_coherence(np.sin(2 * np.pi * t / 16), np.sin(2 * np.pi * t / 16 + 0.2), 1000.0)
    values = np.asarray(result["magnitude_squared_coherence"])
    frequencies = np.asarray(result["coherence_frequency_Hz"])
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.diff(frequencies) >= 0.0)
    # For the same deterministic signal, magnitude-squared coherence is one
    # wherever the estimate has nonzero power (up to floating-point tolerance).
    identical = magnitude_squared_coherence(np.sin(2 * np.pi * t / 16), np.sin(2 * np.pi * t / 16), 1000.0)
    assert np.max(identical["magnitude_squared_coherence"]) == pytest.approx(1.0, abs=1e-12)


def test_faraday_induction_rejects_boolean_turn_count():
    with pytest.raises(ValueError, match="not a boolean"):
        faraday_induction(1.0, True)


def test_bolometry_inverse_reconstruction_is_explicitly_model_limited():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0])
    result = bolometry_regularized_inverse(y, A, 0.1, sigma_measurements=np.ones(3) * 0.01)
    assert len(result["emissivity_estimate"]) == 2
    assert len(result["uncertainty_emissivity_estimate"]) == 2
    assert "Unconstrained linear Tikhonov estimate" in result["reconstruction_note"]
    assert math.isfinite(result["residual_norm"])

def test_bolometry_tikhonov_solution_matches_diagonal_analytic_reference():
    y = np.array([1.0, 2.0])
    result = bolometry_regularized_inverse(y, np.eye(2), 0.5)
    assert result["emissivity_estimate"] == pytest.approx([0.8, 1.6], rel=1e-12)
    assert result["residual_norm"] == pytest.approx(math.sqrt(0.2), rel=1e-12)
