"""Product/API tests for the physical non-relativistic multi-view Thomson EVDF."""

import numpy as np
import pytest

from tis.products.thomson.api import reconstruct_nonrelativistic_3d_evdf


def _synthetic_multiview_case():
    wavelength = np.linspace(499.85, 500.15, 121)
    laser = 500.0
    angles = np.full(3, np.pi / 2.0)
    directions = np.eye(3)
    velocity = np.array([[0.0, 0.0, 0.0], [1.0e4, 0.0, 0.0]])
    volume = np.ones(2)
    gains = np.ones(3)
    background = np.full((3, wavelength.size), 0.05)

    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    widths = np.diff(edges)

    kernels = np.zeros((3, wavelength.size, velocity.shape[0]))
    q_factor = 2.0 * np.sin(np.pi / 4.0)
    for view, qhat in enumerate(directions):
        for voxel, vector in enumerate(velocity):
            centre = laser * (1.0 - q_factor * float(qhat @ vector) / 299792458.0)
            if centre <= wavelength[0]:
                kernels[view, 0, voxel] = 0.5 / widths[0]
            elif centre >= wavelength[-1]:
                kernels[view, -1, voxel] = 0.5 / widths[-1]
            else:
                right = int(np.searchsorted(wavelength, centre, side="right"))
                left = right - 1
                fraction = (centre - wavelength[left]) / (wavelength[right] - wavelength[left])
                kernels[view, left, voxel] = 0.5 * (1.0 - fraction) / widths[left]
                kernels[view, right, voxel] = 0.5 * fraction / widths[right]

    truth = np.array([0.7, 0.3])
    measured = np.einsum("jiv,v->ji", kernels, truth * volume) + background
    return {
        "wavelength_nm": wavelength,
        "measured_spectra": measured,
        "scattering_vector_unit": directions,
        "velocity_grid_m_s": velocity,
        "laser_wavelength_nm": laser,
        "scattering_angle_rad": angles,
        "view_gain": gains,
        "instrument_response_matrix": np.eye(wavelength.size),
        "background_spectra": background,
        "voxel_volume_m3_s3": volume,
        "regularization_matrix": np.array([[-1.0, 1.0]]),
        "regularization_strength": 1.0e-10,
        "measurement_uncertainty": np.full_like(measured, 0.01),
    }, truth


def test_registered_physical_multiview_evdf_recovers_synthetic_distribution():
    config, truth = _synthetic_multiview_case()
    result = reconstruct_nonrelativistic_3d_evdf(**config)

    assert result["physical_forward_model"] == (
        "elastic_single_particle_unpolarized_nonrelativistic_optical_thomson"
    )
    assert result["status"] == "identified_from_multi_view_measurements"
    assert result["data_only_jacobian_rank"] == 2
    assert result["null_space_dimension"] == 0
    assert result["evdf_m3_s3"] is not None
    np.testing.assert_allclose(
        np.sum(result["evdf_m3_s3"] * config["voxel_volume_m3_s3"]),
        1.0, rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(result["evdf_m3_s3"], truth, rtol=2e-3, atol=2e-3)
    assert result["temperature_tensor_eV"].shape == (3, 3)
    assert result["third_central_velocity_moment_tensor_m3_s3"].shape == (3, 3, 3)
    assert result["heat_flux_per_density_W_m"] is not None


def test_registered_entry_and_product_api_contract_are_consistent():
    from tis.products.catalog import get_product
    from tis.products.thomson.registry import THOMSON_ENTRY_POINTS
    from tokamak_analysis.analysis_registry import get_analysis

    assert THOMSON_ENTRY_POINTS[88] == (
        "advanced_inference", "reconstruct_optical_thomson_3d_evdf"
    )
    assert get_analysis(88)["function"] == "reconstruct_optical_thomson_3d_evdf"
    assert 88 in get_product("thomson-scattering")["entry_ids"]


def test_physical_multiview_evdf_rejects_relativistic_velocity_grid():
    config, _ = _synthetic_multiview_case()
    config["velocity_grid_m_s"] = np.array([[0.0, 0.0, 0.0], [0.11 * 299792458.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="max\(\|v\|\)/c < 0.1"):
        reconstruct_nonrelativistic_3d_evdf(**config)
