import numpy as np
from scipy.constants import mu_0, c
from tokamak_analysis.ece_kinetic_modes import solve_two_mode_kinetic_ece_profile


def test_two_mode_kinetic_ece_profile_connects_opacity_and_transfer():
    f = 45e9
    k0 = 2*np.pi*f/c
    # Vacuum plane-wave basis with each column normalized to one W/m^2.
    amplitude = np.sqrt(2.0*mu_0*c)
    basis = np.array([[amplitude, 0.0], [0.0, amplitude], [0.0, 0.0]], dtype=complex)
    out = solve_two_mode_kinetic_ece_profile(
        [f], [0.2],
        wavevector_rad_m=[[0.0, 0.0, k0]],
        magnetic_field_T=[[0.0, 0.0, 2.0]],
        electron_density_m3=[1e19],
        electron_temperature_eV=[1000.0],
        mode_electric_field_unit_flux=[basis],
        harmonic_order=8, quadrature_order=16,
        damping_rate_rad_s=2e7,
    )
    assert out["absorption_matrix_m_inverse"].shape == (1, 1, 2, 2)
    assert out["modal_coherency_out_K"].shape == (1, 2, 2)
    assert np.all(np.isfinite(out["brightness_temperature_out_K"]))
