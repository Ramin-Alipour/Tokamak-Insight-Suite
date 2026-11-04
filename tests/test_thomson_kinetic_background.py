import numpy as np

from tokamak_analysis.thomson_kinetic_background import (
    fit_noncollective_kinetic_thomson_with_background,
)


def test_kappa_structured_background_fit_retains_joint_cross_covariance():
    wavelength = np.linspace(529.0, 535.0, 181)
    laser = 532.0
    angle = np.pi / 2.0
    response = np.eye(wavelength.size)
    truth = np.array([300.0, 8.0e4, 3.5])
    q = 2.0 * np.sin(angle / 2.0)
    center = laser * (1.0 - q * truth[1] / 299792458.0)
    sigma = laser * 2.0 * np.sin(angle / 2.0) * np.sqrt(truth[0] / 510998.95069)
    df = 2.0 * truth[2] - 1.0
    scale = sigma * np.sqrt((df - 2.0) / df)
    shape = (1.0 + ((wavelength - center) / scale) ** 2 / df) ** (-truth[2])
    z = 2.0 * (wavelength - wavelength.min()) / np.ptp(wavelength) - 1.0
    measured = 2.3 * shape + 0.04 + 0.006 * z
    result = fit_noncollective_kinetic_thomson_with_background(
        wavelength, measured,
        physical_model="kappa",
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        instrument_response=response,
        physical_initial=[260.0, 0.0, 4.2],
        background_model="linear",
        measurement_uncertainty=np.full(wavelength.size, 0.003),
        model_discrepancy_covariance=np.eye(wavelength.size) * 1.0e-6,
        systematic_covariance=np.eye(wavelength.size) * 4.0e-7,
        max_nfev=5000,
    )
    assert result["optimizer_success"]
    assert result["joint_parameter_covariance"].shape == (6, 6)
    assert result["physical_background_cross_covariance"].shape == (3, 2)
    assert result["jacobian_rank"] == result["jacobian_parameter_count"]
    assert np.allclose(result["physical_parameters"], truth, rtol=0.04, atol=1e-3)
    assert result["absolute_density_inferred"] is False
    assert result["model_discrepancy_covariance_included"] is True
    assert result["systematic_covariance_included"] is True
    assert result["covariance_components"] == {
        "measurement_covariance_supplied": True,
        "model_discrepancy_covariance_supplied": True,
        "systematic_covariance_supplied": True,
        "combined_covariance_used_for_whitening": True,
    }
    assert result["noise_model"] == "measurement_plus_model_discrepancy_plus_systematic_covariance"
    assert result["parameter_identifiability"]["status"] == "locally_identifiable_after_joint_structured_background_fit"
    assert np.allclose(
        result["joint_parameter_covariance"],
        result["joint_parameter_covariance"].T,
        rtol=1e-10,
        atol=1e-14,
    )
    assert np.all(np.linalg.eigvalsh(result["joint_parameter_covariance"]) >= -1e-12)
    assert result["cross_section_model"] == "isotropic_finite_variance_Kappa_projection"


def test_structured_background_fit_rejects_non_psd_model_discrepancy_covariance():
    import pytest

    wavelength = np.linspace(529.0, 535.0, 41)
    measured = 1.0 + np.exp(-0.5 * ((wavelength - 532.0) / 0.5) ** 2)
    with pytest.raises(ValueError, match="model_discrepancy_covariance must be positive semidefinite"):
        fit_noncollective_kinetic_thomson_with_background(
            wavelength, measured,
            physical_model="kappa",
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0,
            instrument_response=np.eye(wavelength.size),
            physical_initial=[300.0, 0.0, 3.5],
            model_discrepancy_covariance=-np.eye(wavelength.size),
        )
