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
