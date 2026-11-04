import numpy as np
import pytest

from tokamak_analysis.thomson_anisotropic_inference import (
    fit_gyrotropic_bi_maxwellian_thomson,
)


def _synthetic_spectra():
    wavelength = np.linspace(1062.0, 1066.0, 161)
    laser = 1064.0
    angle = np.pi / 2.0
    b = np.array([0.0, 0.0, 1.0])
    q = np.array([
        [0.0, 0.0, 1.0],
        [np.sqrt(3.0) / 2.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
    ])
    tpar, tperp = 900.0, 250.0
    mu2 = (q @ b) ** 2
    directional = tperp + (tpar - tperp) * mu2
    flows = np.array([1.0e4, -1.5e4, 2.0e4])
    spectra = []
    for i, tq in enumerate(directional):
        sigma_v = np.sqrt(1.602176634e-19 * tq / 9.1093837015e-31)
        center = laser * (1.0 - 2.0*np.sin(angle/2.0)*flows[i]/299792458.0)
        sigma_lam = laser * 2.0*np.sin(angle/2.0)*sigma_v/299792458.0
        profile = np.exp(-0.5*((wavelength-center)/sigma_lam)**2)/(np.sqrt(2*np.pi)*sigma_lam)
        spectra.append((0.03 + 0.8*profile + 1.0e-4*(wavelength-laser)))
    return wavelength, np.asarray(spectra), b, q


def test_joint_anisotropic_optical_fit_recovers_parallel_and_perpendicular_temperatures():
    wavelength, spectra, b, q = _synthetic_spectra()
    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, spectra, laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2,
        magnetic_field_unit=b, scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=700.0,
        initial_temperature_perpendicular_eV=350.0,
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    assert result["reportable_as_inferred_parameter"] is True
    assert result["electron_temperature_parallel_eV"] == pytest.approx(900.0, rel=2e-2)
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(250.0, rel=2e-2)
    assert result["electron_velocity_projection_m_s"] == pytest.approx([1.0e4, -1.5e4, 2.0e4], abs=2.0e3)
    assert np.max(np.abs(result["residual_spectra"])) < 1e-5
    assert result["model_validity"]["relativistic_corrections"] is False


def test_single_pitch_angle_does_not_report_separate_anisotropy_temperatures():
    wavelength, spectra, b, q = _synthetic_spectra()
    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, spectra[:1], laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2,
        magnetic_field_unit=b, scattering_vector_unit=q[:1],
        instrument_response_matrix=np.eye(wavelength.size),
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    assert result["anisotropy_identifiability"] == "non_identifiable"
    assert result["electron_temperature_parallel_eV"] is None
    assert result["electron_temperature_perpendicular_eV"] is None
    assert np.isfinite(result["directional_temperature_eV"][0])


def test_anisotropic_fit_rejects_non_normalized_response_and_bad_geometry():
    wavelength, spectra, b, q = _synthetic_spectra()
    response = np.eye(wavelength.size)
    response[:, 0] *= 0.5
    with pytest.raises(ValueError, match="columns must sum to one"):
        fit_gyrotropic_bi_maxwellian_thomson(
            wavelength, spectra, laser_wavelength_nm=1064.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
            scattering_vector_unit=q, instrument_response_matrix=response,
        )
    with pytest.raises(ValueError, match="unit norm"):
        fit_gyrotropic_bi_maxwellian_thomson(
            wavelength, spectra, laser_wavelength_nm=1064.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=[0, 0, 2],
            scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
        )
