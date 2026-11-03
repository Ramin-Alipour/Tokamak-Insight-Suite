import numpy as np

from tokamak_analysis.thomson_gap_closure import (
    reconstruct_psf_aware_thomson_profile,
    reconstruct_relativistic_projected_thomson_evdf,
)


def test_psf_aware_profile_recovers_full_rank_latent_profile():
    mixing = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    truth = np.array([2.0, 1.0])
    result = reconstruct_psf_aware_thomson_profile(
        mixing @ truth, mixing, regularization_strength=1e-10,
        measurement_uncertainty=np.full(3, 1e-3),
    )
    assert result["status"] == "identified_from_measurement"
    np.testing.assert_allclose(result["latent_profile"], truth, rtol=2e-3, atol=2e-3)
    assert result["null_space_dimension"] == 0


def test_psf_aware_profile_does_not_report_null_space_as_primary():
    mixing = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    result = reconstruct_psf_aware_thomson_profile(
        np.array([2.0, 2.0]), mixing, regularization_strength=1e-2,
    )
    assert result["latent_profile"] is None
    assert result["status"] == "conditional_regularized_candidate"
    assert result["null_space_dimension"] >= 1


def test_relativistic_projected_evdf_normalizes_and_reports_model_scope():
    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    angle = np.pi / 2
    velocity = np.linspace(-3.0e6, 3.0e6, 121)
    beta = velocity / 299792458.0
    centres = laser * (1.0 - beta * np.sin(angle / 2)) / (1.0 + beta * np.sin(angle / 2))
    widths = np.diff(np.r_[wavelength[0]-(wavelength[1]-wavelength[0])/2,
                           0.5*(wavelength[:-1]+wavelength[1:]),
                           wavelength[-1]+(wavelength[-1]-wavelength[-2])/2])
    latent = np.zeros((wavelength.size, velocity.size))
    for j, centre in enumerate(centres):
        right = np.searchsorted(wavelength, centre, side="right")
        left = right - 1
        frac = (centre-wavelength[left])/(wavelength[right]-wavelength[left])
        latent[left,j] = (1-frac)/widths[left]
        latent[right,j] = frac/widths[right]
    dv_weights = np.full(velocity.size, velocity[1] - velocity[0])
    dv_weights[[0, -1]] *= 0.5
    forward = latent * dv_weights[None, :]
    truth = np.exp(-0.5*(velocity/7e5)**2)
    truth /= np.trapz(truth, velocity)
    measured = forward @ truth
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, measured, laser_wavelength_nm=laser,
        scattering_angle_rad=angle, instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(measured), velocity_grid_m_s=velocity,
        regularization_strength=1e-3,
    )
    assert result["status"] in {"completed", "conditional_regularized_candidate"}
    candidate = result["regularized_candidate_evdf_s_m"]
    assert np.isclose(np.trapz(candidate, velocity), 1.0, rtol=1e-8)
    if result["status"] == "completed":
        assert result["projected_evdf_s_m"] is not None
    else:
        assert result["projected_evdf_s_m"] is None
    assert result["absolute_density_inferred"] is False
    assert "Compton recoil" in " ".join(result["assumptions"])


def test_velocity_quadrature_is_included_and_grid_integral_is_resolution_consistent():
    from tokamak_analysis.thomson_gap_closure import _trapezoid_quadrature_weights

    coarse = np.linspace(-2.0e6, 2.0e6, 101)
    fine = np.linspace(-2.0e6, 2.0e6, 401)
    coarse_weights = _trapezoid_quadrature_weights(coarse, "velocity_grid_m_s")
    fine_weights = _trapezoid_quadrature_weights(fine, "velocity_grid_m_s")
    coarse_integral = np.sum(np.ones_like(coarse) * coarse_weights)
    fine_integral = np.sum(np.ones_like(fine) * fine_weights)
    np.testing.assert_allclose(coarse_integral, 4.0e6, rtol=1e-12)
    np.testing.assert_allclose(fine_integral, coarse_integral, rtol=1e-12)
    assert coarse_weights[0] == 0.5 * (coarse[1] - coarse[0])
    assert coarse_weights[-1] == coarse_weights[0]
