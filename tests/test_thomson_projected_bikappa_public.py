"""Physical and public-API regressions for the projected gyrotropic bi-Kappa EVDF."""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
from tokamak_analysis.thomson_kinetic_models import project_gyrotropic_bi_kappa_electron_vdf


def _inputs(mu):
    return {
        "velocity_grid_m_s": np.linspace(-2.0e8, 2.0e8, 12001),
        "temperature_parallel_eV": 1500.0,
        "temperature_perpendicular_eV": 700.0,
        "kappa": 4.0,
        "magnetic_field_unit": [np.sqrt(1.0-mu*mu), 0.0, mu],
        "scattering_vector_unit": [0.0, 0.0, 1.0],
        "flow_projection_m_s": 1.2e5,
    }


@pytest.mark.parametrize("mu", [0.0, 0.5, 1.0])
def test_public_projected_bikappa_recovers_directional_second_moment(mu):
    inputs = _inputs(mu)
    result = run_advanced_thomson_physics("thomson_projected_bi_kappa_evdf", **inputs)
    expected_temperature = 700.0 + (1500.0-700.0)*mu**2
    velocity = result["velocity_grid_m_s"]
    pdf = result["projected_vdf_s_m"]
    mean = trapezoid(velocity*pdf, velocity)
    variance = np.trapezoid((velocity-mean)**2*pdf, velocity)

    assert result["model"] == "gyrotropic_finite_variance_bi_kappa_exact_projected_marginal"
    assert result["projected_temperature_eV"] == pytest.approx(expected_temperature, rel=1e-13)
    assert result["projected_velocity_mean_m_s"] == pytest.approx(1.2e5)
    assert np.trapezoid(pdf, velocity) == pytest.approx(1.0, rel=2e-3)
    assert mean == pytest.approx(1.2e5, abs=2.0e4)
    assert variance == pytest.approx(1.602176634e-19*expected_temperature/9.1093837015e-31, rel=3e-2)
    assert result["separate_parallel_perpendicular_temperatures_identifiable"] is False


def test_projected_bikappa_rejects_infinite_variance_domain():
    inputs = _inputs(0.5)
    inputs["kappa"] = 1.5
    with pytest.raises(ValueError, match="kappa > 3/2"):
        run_advanced_thomson_physics("thomson_projected_bi_kappa_evdf", **inputs)


def test_projected_bikappa_public_api_matches_physical_kernel():
    inputs = _inputs(0.6)
    public = run_advanced_thomson_physics("thomson_projected_bi_kappa_evdf", **inputs)
    direct = project_gyrotropic_bi_kappa_electron_vdf(**inputs)
    np.testing.assert_allclose(public["projected_vdf_s_m"], direct["projected_vdf_s_m"])
    assert public["velocity_grid_probability_coverage"] == pytest.approx(
        direct["velocity_grid_probability_coverage"], rel=1e-13
    )
