import numpy as np
import pytest

from tis.physics.shared.thomson_evdf import (
    _projected_evdf_forward_matrix,
    _velocity_quadrature_weights,
)
from tis.physics.shared.thomson_evdf_poisson import reconstruct_thomson_evdf_poisson


def _synthetic_counts():
    wavelength = np.linspace(499.5, 500.5, 101)
    velocity = np.linspace(-2.0e5, 2.0e5, 81)
    laser = 500.0
    angle = np.pi / 2.0
    response = np.eye(wavelength.size)
    kernel, _ = _projected_evdf_forward_matrix(
        wavelength, velocity, response, laser,
        2.0 * np.sin(angle / 2.0),
    )
    weights = _velocity_quadrature_weights(velocity)
    sigma_v = 5.0e4
    truth = np.exp(-0.5 * (velocity / sigma_v) ** 2)
    truth /= weights @ truth
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    widths = np.diff(edges)
    background = np.full(wavelength.size, 4.0)
    expected = 2.0e6 * (widths * (kernel @ truth)) + background
    counts = np.random.default_rng(1729).poisson(expected).astype(int)
    return wavelength, velocity, counts, background, response, truth


def test_poisson_inverse_recovers_normalized_projected_evdf_from_synthetic_counts():
    wavelength, velocity, counts, background, response, truth = _synthetic_counts()
    result = reconstruct_thomson_evdf_poisson(
        wavelength, counts,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=response,
        background_expected_counts=background,
        velocity_grid_m_s=velocity,
        regularization_strength=1.0e-4,
        max_iter=5000,
    )
    density = result["evdf_parallel_s_per_m"]
    weights = _velocity_quadrature_weights(velocity)
    assert result["optimizer_success"] is True
    assert np.all(density >= 0.0)
    assert weights @ density == pytest.approx(1.0, rel=1e-7, abs=1e-9)
    assert result["likelihood"] == "independent_poisson_counts"
    assert result["physical_scope"] == "one_dimensional_q_parallel_velocity_marginal_only"
    assert result["poisson_deviance"] < 250.0
    assert np.corrcoef(density, truth)[0, 1] > 0.90
    assert result["fitted_expected_counts"].shape == counts.shape
    assert result["expected_count_equation"].startswith("mu_i=A_tot")
    assert result["absolute_density_inferred"] is False


def test_poisson_inverse_rejects_noninteger_counts_and_invalid_normalization_inputs():
    wavelength, velocity, counts, background, response, _ = _synthetic_counts()
    common = dict(
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=response,
        background_expected_counts=background,
        velocity_grid_m_s=velocity,
        regularization_strength=1.0e-4,
    )
    with pytest.raises(ValueError, match="integer detector-bin counts"):
        reconstruct_thomson_evdf_poisson(wavelength, counts.astype(float) + 0.25, **common)
    with pytest.raises(ValueError, match="regularization_strength"):
        reconstruct_thomson_evdf_poisson(wavelength, counts, **{**common, "regularization_strength": 0.0})
    with pytest.raises(ValueError, match="background_expected_counts"):
        reconstruct_thomson_evdf_poisson(
            wavelength, counts, **{**common, "background_expected_counts": -np.ones_like(background)}
        )


def test_poisson_evdf_is_reachable_through_product_api_and_advanced_dashboard_dispatch():
    from tis.products.thomson.api import reconstruct_projected_evdf_poisson
    from tis.products.thomson.api import run_advanced_physics, run_analysis
    from tis.products.catalog import THOMSON_ENTRY_POINTS, get_product
    from tokamak_analysis.analysis_registry import resolve_analysis
    from tis.ui.advanced_thomson_physics_page import MODEL_LABELS, MODEL_PHYSICS_NOTES

    wavelength, velocity, counts, background, response, _ = _synthetic_counts()
    kwargs = dict(
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=response,
        background_expected_counts=background,
        velocity_grid_m_s=velocity,
        regularization_strength=1.0e-4,
        max_iter=5000,
    )
    direct = reconstruct_projected_evdf_poisson(wavelength, counts, **kwargs)
    dispatched = run_advanced_physics(
        "thomson_projected_evdf_poisson", wavelength, counts, **kwargs
    )
    registered = run_analysis(89, wavelength, counts, **kwargs)
    assert 89 in get_product("thomson-scattering")["entry_ids"]
    assert THOMSON_ENTRY_POINTS[89] == (
        "thomson_evdf_poisson", "reconstruct_thomson_evdf_poisson"
    )
    assert resolve_analysis(89).__name__ == "reconstruct_thomson_evdf_poisson"
    assert direct["likelihood"] == dispatched["likelihood"] == registered["likelihood"] == "independent_poisson_counts"
    assert "projected_evdf_physical_moments" in direct
    assert direct["projected_evdf_physical_moments"]["central_third_velocity_moment_m3_s3"] is not None
    assert direct["projected_evdf_physical_moments"]["parallel_projected_heat_flux_W_m2"] is None
    assert direct["projected_evdf_physical_moments"]["heat_flux_physical_scope"] == (
        "one_dimensional_parallel_kinetic_energy_flux_component_only; "
        "not_the_full_three_dimensional_electron_heat_flux"
    )
    assert "thomson_projected_evdf_poisson" in MODEL_LABELS
    assert "thomson_projected_evdf_poisson" in MODEL_PHYSICS_NOTES


def test_poisson_inverse_propagates_moment_covariance_and_optional_projected_heat_flux():
    wavelength, velocity, counts, background, response, _ = _synthetic_counts()
    common = dict(
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=response,
        background_expected_counts=background,
        velocity_grid_m_s=velocity,
        regularization_strength=1.0e-4,
        max_iter=5000,
    )
    without_density = reconstruct_thomson_evdf_poisson(wavelength, counts, **common)
    moments = without_density["projected_evdf_physical_moments"]
    assert moments["directional_equivalent_temperature_eV"] > 0.0
    assert moments["moment_covariance_order"][4] == "central_third_velocity_moment_m3_s3"
    assert moments["parallel_projected_heat_flux_W_m2"] is None
    if moments["moment_covariance"] is not None:
        covariance = moments["moment_covariance"]
        assert covariance.shape == (8, 8)
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-9, atol=1e-30)
        scale = max(np.linalg.norm(covariance, 2), 1e-300)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-8 * scale

    with_density = reconstruct_thomson_evdf_poisson(
        wavelength, counts, **common,
        electron_density_m3=2.0e19,
        electron_density_uncertainty_m3=1.0e17,
    )
    flux = with_density["projected_evdf_physical_moments"]
    expected_flux = (
        0.5 * 9.1093837139e-31 * 2.0e19
        * flux["central_third_velocity_moment_m3_s3"]
    )
    assert flux["parallel_projected_heat_flux_W_m2"] == pytest.approx(expected_flux, rel=1e-12, abs=1e-30)
    assert flux["parallel_projected_heat_flux_standard_uncertainty_W_m2"] is not None
    assert flux["full_three_dimensional_heat_flux_identified"] is False


def test_poisson_projected_heat_flux_rejects_unphysical_density_cross_covariance():
    wavelength, velocity, counts, background, response, _ = _synthetic_counts()
    with pytest.raises(ValueError, match="requires density"):
        reconstruct_thomson_evdf_poisson(
            wavelength, counts,
            laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2.0,
            instrument_response_matrix=response,
            background_expected_counts=background,
            velocity_grid_m_s=velocity,
            regularization_strength=1.0e-4,
            electron_density_uncertainty_m3=1.0e17,
        )
