import numpy as np
import pytest

from tokamak_analysis.cts_full_em import build_full_em_cts_physical_background


def _case():
    shifts = np.linspace(-2.0e6, 2.0e6, 9)
    probe = 140.0e9
    response = np.eye(shifts.size)
    inputs = {
        "path_length_m": np.array([0.10, 0.15]),
        "electron_temperature_eV": np.array([1200.0, 1400.0]),
        "electron_density_m3": np.array([2.0e19, 1.8e19]),
        "ion_densities_m3": np.array([[2.0e19], [1.8e19]]),
        "ion_charge_states": np.array([1.0]),
        "gaunt_factor": 1.1,
        "ece_absorption_coefficient_m_inverse": np.full((2, shifts.size), 2.0e-4),
        "ece_lte": True,
        "incident_radiance_W_m2_sr_Hz": 0.0,
        "receiver_etendue_m2_sr": 2.0e-5,
        "end_to_end_efficiency": 0.4,
    }
    return shifts, probe, response, inputs


def test_full_em_physical_background_uses_absolute_frequency_and_response_once():
    shifts, probe, response, inputs = _case()
    spectrum, diagnostics = build_full_em_cts_physical_background(shifts, probe, response, radiative_transfer=inputs)
    assert spectrum.shape == shifts.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum > 0.0)
    np.testing.assert_allclose(diagnostics["absolute_receiver_frequency_Hz"], probe + shifts)
    assert diagnostics["instrument_response_applied_once"] is True
    assert diagnostics["ece_model_status"].startswith("mode-resolved")
    np.testing.assert_allclose(diagnostics["response_convolved_receiver_spectral_power_W_Hz"], spectrum)


def test_full_em_physical_background_convolves_with_measured_response():
    shifts, probe, _, inputs = _case()
    response = np.eye(shifts.size)
    response[0, 0] = 0.8
    response[1, 0] = 0.2
    spectrum, diagnostics = build_full_em_cts_physical_background(shifts, probe, response, radiative_transfer=inputs)
    raw = diagnostics["pre_response_receiver_spectral_power_W_Hz"]
    edges = np.empty(shifts.size + 1)
    edges[1:-1] = 0.5 * (shifts[:-1] + shifts[1:])
    edges[0] = shifts[0] - 0.5 * (shifts[1] - shifts[0])
    edges[-1] = shifts[-1] + 0.5 * (shifts[-1] - shifts[-2])
    widths = np.diff(edges)
    np.testing.assert_allclose(spectrum, (response @ (raw * widths)) / widths)


def test_full_em_physical_background_requires_explicit_ece_choice():
    shifts, probe, response, inputs = _case()
    incomplete = dict(inputs)
    incomplete.pop("ece_absorption_coefficient_m_inverse")
    with pytest.raises(ValueError, match="missing"):
        build_full_em_cts_physical_background(shifts, probe, response, radiative_transfer=incomplete)


def test_full_em_physical_background_rejects_nonpositive_absolute_frequency():
    shifts, probe, _, inputs = _case()
    with pytest.raises(ValueError, match="must remain positive"):
        build_full_em_cts_physical_background(np.array([-2.0 * probe, -probe]), probe, np.eye(2), radiative_transfer=inputs)



def test_full_em_physical_background_is_exposed_by_dashboard_contract():
    from pathlib import Path

    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "cts_full_physical_background_json" in app
    assert "cts_full_physical_background_covariance_upload" in app
    assert "physical_background_radiative_transfer" in app
    assert "physical_background_covariance_W2_per_Hz2" in app
