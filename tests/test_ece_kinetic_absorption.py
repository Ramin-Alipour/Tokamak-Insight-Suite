import numpy as np

from tokamak_analysis.ece_analysis import kinetic_ece_absorption


def test_kinetic_ece_absorption_returns_local_coefficients():
    # Off-resonant Maxwellian case: finite local conductivity and transfer data.
    f = 45e9
    omega = 2*np.pi*f
    k0 = omega/299792458.0
    result = kinetic_ece_absorption(
        f, [0.0, 0.0, k0], [0.0, 0.0, 2.0],
        1e19, 1000.0, [1.0, 0.0, 0.0],
        harmonic_order=8, quadrature_order=12,
        damping_rate_rad_s=2e7,
    )
    assert result["absorption_coefficient_m_inverse"] >= 0.0
    assert np.isfinite(result["emission_coefficient_W_m3_sr_Hz"])
    assert result["conductivity_S_m"].shape == (3, 3)
    assert result["kinetic_model"].startswith("magnetized_nonrelativistic")


def test_kinetic_ece_absorption_rejects_nonphysical_wavevector():
    with np.testing.assert_raises_regex(ValueError, "wavevector_rad_m"):
        kinetic_ece_absorption(
            45e9, [0.0, 0.0, 0.0], [0.0, 0.0, 2.0],
            1e19, 1000.0, [1.0, 0.0, 0.0],
            damping_rate_rad_s=1e7,
        )
