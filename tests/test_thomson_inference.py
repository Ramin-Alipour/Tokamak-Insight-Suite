import numpy as np
import pytest

from tokamak_analysis.thomson_inference import analyze_thomson_spectra


def _spectral_case(te_ev, center_nm=532.0, laser_nm=532.0, angle=np.pi/2, n=801):
    x = np.linspace(laser_nm-40.0, laser_nm+40.0, n)
    sigma = 2.0*laser_nm*np.sqrt(te_ev/510998.95069)*np.sin(angle/2.0)
    y = 1200.0*np.exp(-0.5*((x-center_nm)/sigma)**2)+25.0+0.1*(x-center_nm)
    return x, y, sigma


def _asymmetric_lsf(n):
    response = np.zeros((n,n))
    for j in range(n):
        response[j,j] += 0.65
        if j > 0:
            response[j-1,j] += 0.25
        if j < n-1:
            response[j+1,j] += 0.10
    response /= response.sum(axis=0, keepdims=True)
    return response


def test_multi_channel_spectra_recover_channel_resolved_temperature_and_radius():
    x, y1, _ = _spectral_case(100.0)
    _, y2, _ = _spectral_case(150.0)
    out = analyze_thomson_spectra(
        x, np.vstack([y1,y2]), [0.12,0.24],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2,
    )
    assert out["channel_count"] == 2
    np.testing.assert_allclose(out["radius_m"], [0.12,0.24])
    np.testing.assert_allclose(out["electron_temperature_profile_eV"], [100.0,150.0], rtol=0.04)
    assert np.isnan(out["electron_density_profile_m3"]).all()
    assert out["channels"][0]["density_status"] == "not_identifiable_from_spectral_shape_alone"


def test_non_gaussian_instrument_response_matrix_is_used_in_forward_fit():
    x, intrinsic, _ = _spectral_case(100.0)
    response = _asymmetric_lsf(x.size)
    measured = response @ (intrinsic - 25.0) + 25.0
    out = analyze_thomson_spectra(
        x, measured[None,:], [0.2], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=response,
    )
    assert out["channels"][0]["instrument_response_used"] is True
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(100.0, rel=0.05)


def test_absolute_density_requires_explicit_watt_per_nm_calibration():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="spectral_power_density_calibrated"):
        analyze_thomson_spectra(
            x, (y*1e-18)[None,:], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, laser_power_w=2.0,
            interaction_length_m=0.1, collection_solid_angle_sr=1e-3,
            optical_efficiency=0.5, incident_polarization=[1,0,0],
            scattered_direction=[0,1,0],
        )


def test_absolute_density_and_uncertainty_are_reported_when_full_geometry_is_supplied():
    x, y, _ = _spectral_case(100.0)
    out = analyze_thomson_spectra(
        x, (y*1e-18)[None,:], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, laser_power_w=2.0,
        laser_power_uncertainty_W=0.02, interaction_length_m=0.1,
        interaction_length_uncertainty_m=0.001, collection_solid_angle_sr=1e-3,
        collection_solid_angle_uncertainty_sr=1e-5, optical_efficiency=0.5,
        optical_efficiency_uncertainty=0.01, incident_polarization=[1,0,0],
        scattered_direction=[0,1,0], spectral_power_density_calibrated=True,
    )
    channel = out["channels"][0]
    assert channel["electron_density_m3"] > 0
    assert channel["electron_density_uncertainty_m3"] > 0
    assert channel["density_method"].startswith("absolute integrated")


def test_collective_mode_never_runs_without_explicit_ion_species():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="ion species"):
        analyze_thomson_spectra(
            x, y[None,:], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, model="collective_maxwellian",
        )


def test_channel_specific_wavelength_axes_are_supported():
    x1, y1, _ = _spectral_case(100.0)
    x2, y2, _ = _spectral_case(150.0, center_nm=532.02)
    out = analyze_thomson_spectra(
        np.vstack([x1,x2]), np.vstack([y1,y2]), [0.1,0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2,
    )
    assert out["channel_count"] == 2
    assert out["channels"][1]["fitted_center_nm"] == pytest.approx(532.02, abs=0.02)
