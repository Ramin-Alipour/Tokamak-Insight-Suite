import numpy as np

import tokamak_analysis.thomson_workflow as workflow


def test_spectral_profile_workflow_exposes_downstream_psf_inverse(monkeypatch):
    radii = np.array([0.1, 0.2, 0.3])
    values = np.array([100.0, 90.0, 80.0])
    monkeypatch.setattr(workflow, "analyze_thomson_spectra", lambda *a, **k: {
        "radius_m": radii,
        "model": "non_collective_maxwellian",
        "channels": [],
        "electron_temperature_profile_eV": values,
        "electron_temperature_uncertainty_profile_eV": np.full(3, 2.0),
        "electron_density_profile_m3": np.array([2.0e19, 1.8e19, 1.6e19]),
        "electron_density_uncertainty_profile_m3": np.full(3, 1.0e18),
        "electron_velocity_projection_profile_m_s": np.array([1.0e4, 0.0, -1.0e4]),
        "electron_velocity_projection_uncertainty_profile_m_s": np.full(3, 500.0),
    })
    # A mildly mixing, full-rank channel-to-profile operator.
    mixing = np.array([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    result = workflow.reconstruct_thomson_profile_from_spectra(
        np.linspace(531.0, 533.0, 9), np.ones((3, 9)), radii,
        model="non_collective_maxwellian", laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(9),
        spatial_mixing_matrix=mixing, profile_grid_radius_m=radii,
    )
    assert result["psf_aware_profile_reconstruction"]["status"] == "completed"
    assert result["psf_aware_profile_reconstruction"]["profiles"]["electron_temperature"]["analysis_status"] == "completed"


def test_psf_profile_path_requires_explicit_profile_grid(monkeypatch):
    radii = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(workflow, "analyze_thomson_spectra", lambda *a, **k: {
        "radius_m": radii, "model": "non_collective_maxwellian",
        "channels": [], "electron_temperature_profile_eV": np.array([100.0, 90.0, 80.0]),
    })
    import pytest
    with pytest.raises(ValueError, match="profile_grid_radius_m is required"):
        workflow.reconstruct_thomson_profile_from_spectra(
            np.linspace(531.0, 533.0, 9), np.ones((3, 9)), radii,
            model="non_collective_maxwellian", laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(9),
            spatial_mixing_matrix=np.eye(3),
        )
