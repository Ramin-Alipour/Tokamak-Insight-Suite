import numpy as np
import pytest

from tokamak_analysis.thomson_joint_spatial_spectral import reconstruct_joint_thomson_spatial_spectra


def _synthetic_case():
    wavelength = np.linspace(520.0, 544.0, 241)
    radius = np.linspace(0.0, 1.0, 5)
    mixing = np.array([
        [0.80, 0.20, 0.00, 0.00, 0.00],
        [0.15, 0.70, 0.15, 0.00, 0.00],
        [0.00, 0.15, 0.70, 0.15, 0.00],
        [0.00, 0.00, 0.15, 0.70, 0.15],
        [0.00, 0.00, 0.00, 0.20, 0.80],
    ])
    te = np.array([70., 85., 100., 115., 130.])
    flow = np.array([-1.0e4, -0.5e4, 0., 0.5e4, 1.0e4])
    amp = np.array([1., 1.1, 1.2, 1.1, 1.])
    sigma = 2*532*np.sqrt(te/510998.95069)*np.sin(np.pi/4)
    center = 532*(1-2*np.sin(np.pi/4)*flow/299792458.)
    intrinsic = amp[:,None]*np.exp(-0.5*((wavelength[None,:]-center[:,None])/sigma[:,None])**2)
    spectra = mixing @ intrinsic
    spectra += 0.01
    return wavelength, spectra, radius, mixing, te, flow


def test_joint_spatial_spectral_inverse_recovers_synthetic_te_and_flow_profiles():
    x, y, r, h, te, flow = _synthetic_case()
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2,
        signal_uncertainty=np.full_like(y, 0.01), regularization_strength=0.0,
    )
    assert out["analysis_status"] == "completed"
    np.testing.assert_allclose(out["electron_temperature_profile_eV"], te, rtol=0.04)
    np.testing.assert_allclose(out["electron_velocity_projection_profile_m_s"], flow, atol=2.5e3)
    assert out["fitted_spectra"].shape == y.shape
    assert out["parameter_covariance"].shape == (3*r.size+2*y.shape[0],)*2
    assert out["spatial_mixing_operator"] == "researcher_supplied_fixed_forward_operator"
    assert out["experimental_validation"] == "not_performed; numerical software inference only"


def test_joint_inverse_accepts_full_channel_major_covariance():
    x, y, r, h, *_ = _synthetic_case()
    covariance = np.eye(y.size) * 1e-4
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2,
        joint_spectral_covariance=covariance, regularization_strength=1e-5,
    )
    assert out["noise_model"] == "full_channel_major_joint_spectral_covariance"
    assert np.all(np.isfinite(out["electron_temperature_uncertainty_eV"]))


def test_joint_inverse_rejects_singular_mixing_and_invalid_covariance():
    x, y, r, h, *_ = _synthetic_case()
    bad = h.copy()
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="rank deficient"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, bad, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2,
        )
    with pytest.raises(ValueError, match="positive definite"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2,
            joint_spectral_covariance=np.zeros((y.size,y.size)),
        )
