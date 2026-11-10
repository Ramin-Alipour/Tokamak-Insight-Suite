import numpy as np
import pytest
from tis.products.thomson.physics.thomson_distribution import _projected_evdf_physical_moments


def _gaussian_projected_evdf():
    velocity=np.linspace(-6.0e5,6.0e5,601); sigma=1.0e5
    pdf=np.exp(-0.5*(velocity/sigma)**2); pdf/=np.trapz(pdf,velocity)
    return velocity,pdf


def test_projected_parallel_pressure_uses_second_central_evdf_moment():
    velocity,pdf=_gaussian_projected_evdf(); density=2.5e19
    result=_projected_evdf_physical_moments(velocity,pdf,np.eye(velocity.size)*1e-20,electron_density_m3=density,electron_density_uncertainty_m3=0.0)
    expected=9.1093837139e-31*density*result["velocity_parallel_variance_m2_s2"]
    assert result["parallel_projected_pressure_Pa"]==pytest.approx(expected,rel=1e-12)
    assert result["parallel_pressure_uncertainty_assumption"]=="first_order_independent_density_and_evdf_variance"
    assert result["parallel_projected_pressure_standard_uncertainty_Pa"] is not None
    assert result["equations"]["parallel_projected_pressure"].startswith("P_qq = m_e n_e")
    assert result["heat_flux_identifiability"]["status"]=="not_identified_from_single_projected_marginal"


def test_projected_parallel_pressure_is_not_reported_without_independent_density():
    velocity,pdf=_gaussian_projected_evdf(); result=_projected_evdf_physical_moments(velocity,pdf)
    assert result["parallel_projected_pressure_Pa"] is None
    assert result["parallel_projected_pressure_standard_uncertainty_Pa"] is None


def test_projected_parallel_pressure_propagates_density_variance_cross_covariance():
    velocity,pdf=_gaussian_projected_evdf(); covariance=np.eye(velocity.size)*1e-20
    density=2e19; density_sigma=1e17
    base=_projected_evdf_physical_moments(velocity,pdf,covariance,electron_density_m3=density,electron_density_uncertainty_m3=density_sigma)
    vi=base["moment_covariance_order"].index("velocity_parallel_variance_m2_s2")
    varvar=base["moment_covariance"][vi,vi]; cross=0.25*density_sigma*np.sqrt(varvar)
    result=_projected_evdf_physical_moments(velocity,pdf,covariance,electron_density_m3=density,electron_density_uncertainty_m3=density_sigma,electron_density_variance_cross_covariance_m_minus1_s_minus2=cross)
    mass=9.1093837139e-31; variance=result["velocity_parallel_variance_m2_s2"]
    expected=mass**2*(density**2*varvar+variance**2*density_sigma**2+2*density*variance*cross)
    assert result["parallel_projected_pressure_standard_uncertainty_Pa"]==pytest.approx(np.sqrt(expected),rel=1e-10)
    assert result["parallel_pressure_uncertainty_assumption"]=="joint_density_evdf_variance_covariance"


def test_projected_parallel_pressure_rejects_cross_covariance_without_independent_density():
    velocity,pdf=_gaussian_projected_evdf()
    with pytest.raises(ValueError,match="requires an independently supplied"):
        _projected_evdf_physical_moments(velocity,pdf,np.eye(velocity.size),electron_density_variance_cross_covariance_m_minus1_s_minus2=1.0)


def test_projected_parallel_pressure_marks_unsupplied_density_uncertainty_as_exact_input():
    velocity, pdf = _gaussian_projected_evdf()
    result = _projected_evdf_physical_moments(
        velocity, pdf, np.eye(velocity.size) * 1.0e-20,
        electron_density_m3=2.0e19,
    )
    assert result["parallel_projected_pressure_Pa"] is not None
    assert result["parallel_pressure_uncertainty_assumption"] == (
        "evdf_variance_only_density_uncertainty_not_supplied"
    )


def test_projected_pressure_is_exposed_through_specialist_product_api():
    from tis.products.thomson.api import analyze_spectral_distribution

    wavelength = np.linspace(520.0, 544.0, 241)
    background = np.full_like(wavelength, 0.02)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 1.0) ** 2) + background
    result = analyze_spectral_distribution(
        wavelength,
        signal,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        signal_uncertainty=np.full_like(wavelength, 1.0e-3),
        projected_evdf_options={
            "velocity_limit_m_s": 1.0e6,
            "velocity_points": 81,
            "regularization_strength": 1.0e-3,
            "electron_density_m3": 2.0e19,
            "electron_density_uncertainty_m3": 1.0e18,
            "electron_density_variance_cross_covariance_m_minus1_s_minus2": 0.0,
        },
    )
    moments = result["projected_evdf_inference"]["projected_evdf_physical_moments"]
    assert moments["parallel_projected_pressure_Pa"] > 0.0
    assert np.isfinite(moments["parallel_projected_pressure_standard_uncertainty_Pa"])
    assert moments["parallel_pressure_uncertainty_assumption"] == (
        "first_order_independent_density_and_evdf_variance"
    )
