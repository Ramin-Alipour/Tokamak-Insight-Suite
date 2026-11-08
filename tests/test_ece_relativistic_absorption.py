import numpy as np
import pytest

from tokamak_analysis.ece_kinetic import relativistic_kinetic_ece_absorption_coefficient


def _inputs():
    frequency = 100e9
    return {
        "frequency_Hz": frequency,
        "wavevector_rad_m": [0.0, 0.0, 2*np.pi*frequency/299792458.0],
        "magnetic_field_T": [0.0, 0.0, 2.0],
        "electron_density_m3": 1e19,
        "electron_distribution": {
            "kind": "relativistic_maxwell_juttner",
            "temperature_eV": 1000.0,
        },
        "polarization": [1.0, 0.0, 0.0],
        "quadrature_order": 4,
        "gyrophase_points": 16,
        "resonance_width_rad_s": 2e7,
    }


def test_relativistic_kinetic_ece_returns_finite_poynting_normalized_absorption():
    result = relativistic_kinetic_ece_absorption_coefficient(**_inputs())
    assert np.isfinite(result["absorption_coefficient_m_inverse"])
    assert result["conductivity_S_m"].shape == (3, 3)
    assert result["relativistic_kinematics"] is True
    assert result["distribution_is_external"] is True
    assert result["noise_model_used_for_absorption"] is False


def test_relativistic_kinetic_ece_rejects_invalid_mode_flux():
    inputs = _inputs()
    inputs["polarization"] = [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="Poynting flux"):
        relativistic_kinetic_ece_absorption_coefficient(**inputs)


def test_relativistic_kinetic_ece_requires_explicit_distribution():
    inputs = _inputs()
    inputs["electron_distribution"] = None
    with pytest.raises((ValueError, TypeError)):
        relativistic_kinetic_ece_absorption_coefficient(**inputs)


def test_relativistic_ece_profile_keeps_signed_absorption_and_declares_no_transfer():
    from tokamak_analysis.ece_kinetic import solve_relativistic_kinetic_ece_profile

    base = _inputs()
    result = solve_relativistic_kinetic_ece_profile(
        [base["frequency_Hz"]],
        [0.1, 0.2],
        wavevector_rad_m=np.array([base["wavevector_rad_m"], base["wavevector_rad_m"]]),
        magnetic_field_T=np.array([base["magnetic_field_T"], base["magnetic_field_T"]]),
        polarization=np.array([base["polarization"], base["polarization"]]),
        electron_density_m3=[1e19, 2e19],
        electron_distributions=[base["electron_distribution"], base["electron_distribution"]],
        quadrature_order=4, gyrophase_points=16,
        resonance_width_rad_s=2e7,
    )
    assert result["absorption_coefficient_m_inverse"].shape == (1, 2)
    assert result["net_gain_mask"].shape == (1, 2)
    assert result["transfer_performed"] is False


def test_relativistic_ece_profile_requires_one_distribution_per_segment():
    from tokamak_analysis.ece_kinetic import solve_relativistic_kinetic_ece_profile

    base = _inputs()
    with pytest.raises(ValueError, match="one explicit EVDF"):
        solve_relativistic_kinetic_ece_profile(
            [base["frequency_Hz"]], [0.1, 0.2],
            wavevector_rad_m=np.array([base["wavevector_rad_m"], base["wavevector_rad_m"]]),
            magnetic_field_T=np.array([base["magnetic_field_T"], base["magnetic_field_T"]]),
            polarization=np.array([base["polarization"], base["polarization"]]),
            electron_density_m3=[1e19, 2e19],
            electron_distributions=[base["electron_distribution"]],
            quadrature_order=4, gyrophase_points=16,
        )
