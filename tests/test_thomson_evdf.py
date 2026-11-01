import numpy as np
import pytest

from tokamak_analysis.thomson_evdf import reconstruct_thomson_evdf


def _case():
    wavelength = np.linspace(499.8, 500.2, 401)
    laser = 500.0
    theta = np.pi / 2
    velocity = 1.8e5
    sigma_v = 2.5e4
    q_over_ki = 2 * np.sin(theta / 2)
    centre = laser * (1 - q_over_ki * velocity / 299_792_458.0)
    sigma_lambda = laser * q_over_ki * sigma_v / 299_792_458.0
    background = np.full(wavelength.size, 3.0)
    signal = 1000 * np.exp(-0.5 * ((wavelength - centre) / sigma_lambda) ** 2) + background
    return wavelength, signal, background


def _run(wavelength, signal, background, **kwargs):
    return reconstruct_thomson_evdf(
        wavelength, signal,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi / 2,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        regularization_strength=1e-3,
        velocity_limit_m_s=2.0e5,
        velocity_points=161,
        **kwargs,
    )


def test_evdf_reconstruction_returns_normalized_nonnegative_1d_projection():
    wavelength, signal, background = _case()
    out = _run(wavelength, signal, background)
    f = out["evdf_parallel_s_per_m"]
    v = out["velocity_parallel_m_s"]
    assert np.all(f >= 0)
    assert np.trapz(f, v) == pytest.approx(1.0, rel=2e-5)
    assert out["scattering_vector_projection_only"] is True
    assert out["absolute_density_identified"] is False
    assert out["directional_equivalent_temperature_eV"] > 0
    assert out["fitted_spectrum"].shape == signal.shape


def test_evdf_requires_explicit_positive_regularization_and_response():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="regularization_strength"):
        reconstruct_thomson_evdf(
            wavelength, signal, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=background, regularization_strength=0.0,
        )
    with pytest.raises(ValueError, match="positive integrated response"):
        reconstruct_thomson_evdf(
            wavelength, signal, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi / 2,
            instrument_response_matrix=np.zeros((wavelength.size, wavelength.size)),
            background_signal=background, regularization_strength=1e-3,
        )


def test_evdf_rejects_incompatible_uncertainty_inputs():
    wavelength, signal, background = _case()
    with pytest.raises(ValueError, match="not both"):
        _run(
            wavelength, signal, background,
            signal_uncertainty=np.ones(signal.size),
            signal_covariance=np.eye(signal.size),
        )


def test_evdf_does_not_claim_a_3d_distribution_or_scalar_temperature():
    wavelength, signal, background = _case()
    out = _run(wavelength, signal, background)
    assert "unrestricted three-dimensional EVDF from one scattering geometry" in out["identifiability"]["non_identifiable"]
    assert "directional_equivalent_temperature_eV" in out
    assert "electron_temperature_eV" not in out
