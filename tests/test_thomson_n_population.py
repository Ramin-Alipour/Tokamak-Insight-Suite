import numpy as np
import pytest

from tokamak_analysis.thomson_n_population import n_population_relativistic_thomson_spectrum


def test_n_population_validates_and_returns_three_component_model():
    x = np.linspace(520.0, 544.0, 81)
    result = n_population_relativistic_thomson_spectrum(
        x, 532.0, np.pi / 2, [300.0, 1800.0, 9000.0], [0.7, 0.2, 0.1],
        flow_projection_m_s=0.0, incident_polarization=[1.0, 0.0, 0.0],
        radial_order=16, angular_order=8,
    )
    assert result['population_count'] == 3
    assert np.isclose(sum(result['population_number_fractions']), 1.0)
    assert result['effective_differential_cross_section_m2_sr'] > 0
    assert np.all(np.isfinite(result['spectral_density_per_nm']))
    assert result['absolute_density_inferred'] is False


def test_n_population_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        n_population_relativistic_thomson_spectrum(
            np.linspace(520.0, 544.0, 81), 532.0, np.pi / 2,
            [500.0, 2000.0], [0.2, 0.2], flow_projection_m_s=0.0,
            incident_polarization=[1.0, 0.0, 0.0], radial_order=16, angular_order=8,
        )


def test_n_population_inverse_fits_ordered_temperatures_and_joint_background():
    from tokamak_analysis.thomson_n_population import (
        fit_n_population_relativistic_thomson_spectrum,
    )

    x = np.linspace(450.0, 614.0, 161)
    response = np.eye(x.size)
    physical = n_population_relativistic_thomson_spectrum(
        x, 532.0, np.pi / 2.0, [120.0, 1200.0], [0.72, 0.28],
        flow_projection_m_s=1.0e4,
        incident_polarization=[1.0, 0.0, 0.0],
        radial_order=16, angular_order=8,
    )
    z = (x - 532.0) / 82.0
    background = 3.0 + 0.2*z
    measured = 8.0e3 * physical["spectral_density_per_nm"] + background
    result = fit_n_population_relativistic_thomson_spectrum(
        x, measured, 532.0, np.pi / 2.0, response,
        population_count=2,
        initial_temperatures_eV=[100.0, 1500.0],
        initial_fractions=[0.7, 0.3],
        initial_flow_projection_m_s=8.0e3,
        incident_polarization=[1.0, 0.0, 0.0],
        signal_uncertainty=np.full(x.size, 0.2),
        background_model="linear",
        radial_order=16, angular_order=8, max_nfev=1000,
    )
    assert result["population_count"] == 2
    assert result["population_temperatures_eV"][0] < result["population_temperatures_eV"][1]
    assert np.sum(result["population_number_fractions"]) == pytest.approx(1.0)
    assert result["joint_parameter_covariance"].shape[0] == len(result["parameter_names"])
    assert result["jacobian_rank"] == result["jacobian_parameter_count"]
    assert result["absolute_density_inferred"] is False
