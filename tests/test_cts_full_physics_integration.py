import numpy as np
from tokamak_analysis.cts_full_physics import (
    analyze_landau_fdt_cts,
    propagate_stratified_nonparaxial_vector_field,
)


def _species():
    return [
        {"name": "electrons", "density_m3": 1.0e18,
         "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
        {"name": "ions", "density_m3": 1.0e18,
         "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19},
    ]


def test_integrated_landau_fdt_cts_returns_screened_spectrum_and_convergence_gate():
    axis = np.linspace(-4.0, 4.0, 3)
    result = analyze_landau_fdt_cts(
        [1.0e5, 2.0e5], axis, axis, axis, _species(),
        common_temperature_J=100.0*1.602176634e-19,
        wave_number_m_inverse=10.0,
        coulomb_logarithm=12.0,
    )
    assert result["dynamic_structure_factor_m3_s"].shape == (2, 2, 2)
    assert result["landau_langevin_spectral_matrix_m3_s"].shape == (2, 2, 2)
    assert result["integration_status"].startswith("landau_operator")
    assert result["quantitative_release_gate"]["velocity_grid_convergence_required"]


def test_stratified_nonparaxial_vector_beam_reduces_to_vacuum_propagation():
    nx = ny = 8
    dx = dy = 0.005
    x = (np.arange(nx) - nx//2)*dx
    y = (np.arange(ny) - ny//2)*dy
    frequency = 10e9
    c0 = 299792458.0
    omega = 2*np.pi*frequency
    k0 = omega/c0
    kx = 2*np.pi/(nx*dx)
    ky = 0.0
    kz = np.sqrt(k0*k0-kx*kx-ky*ky)
    xx, yy = np.meshgrid(x, y)
    phase = np.exp(1j*(kx*xx+ky*yy))
    polarization = np.array([0.0, 1.0, 0.0], dtype=complex)
    polarization /= np.linalg.norm(polarization)
    field = phase[..., None]*polarization
    z = np.array([0.0, 0.01])
    eps = np.repeat(np.eye(3, dtype=complex)[None, :, :], z.size, axis=0)
    result = propagate_stratified_nonparaxial_vector_field(
        x, y, field, z, eps, frequency
    )
    np.testing.assert_allclose(
        result["transmitted_electric_field_V_m"],
        field*np.exp(1j*kz*(z[-1]-z[0])),
        atol=2e-6, rtol=2e-6,
    )
    assert np.linalg.norm(result["reflected_electric_field_V_m"]) < 1e-7
    assert result["paraxial_approximation_used"] is False
    assert result["solved_angular_component_count"] == 1


def test_stratified_nonparaxial_vector_beam_rejects_nonphysical_geometry():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = [0.0, 1.0]
    eps = np.repeat(np.eye(3, dtype=complex)[None, :, :], 2, axis=0)
    field = np.zeros((4, 4, 3), dtype=complex)
    with np.testing.assert_raises_regex(ValueError, "uniformly spaced"):
        propagate_stratified_nonparaxial_vector_field(
            [0.0, 1.0, 2.0, 4.0], y, field, z, eps, 1e9
        )
