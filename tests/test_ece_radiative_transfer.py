import numpy as np
import pytest

from tokamak_analysis.ece_radiative_transfer import (
    planck_spectral_radiance,
    rayleigh_jeans_spectral_radiance,
    solve_ece_radiative_transfer,
)


def test_planck_source_approaches_rayleigh_jeans_at_low_frequency():
    nu = np.array([1.0e6, 2.0e6])
    temp = np.array([1.0e6, 1.0e6])
    assert np.allclose(planck_spectral_radiance(nu, temp),
                       rayleigh_jeans_spectral_radiance(nu, temp), rtol=1e-5)


def test_optically_thick_lte_slab_approaches_planck_source():
    nu = np.array([30.0e9])
    temp = np.array([1.0e6])
    out = solve_ece_radiative_transfer(
        nu, [0.0, 1.0], [100.0], electron_temperature_K=temp
    )
    assert out["spectral_radiance_W_m2_sr_Hz"][0] == pytest.approx(
        planck_spectral_radiance(nu, temp)[0], rel=1e-12
    )
    assert out["optical_depth"][0] == pytest.approx(100.0)


def test_zero_absorption_limit_integrates_explicit_emissivity():
    out = solve_ece_radiative_transfer(
        [10e9], [0.0, 2.0], [0.0],
        emissivity_W_m3_sr_Hz=[3.0],
        incident_radiance_W_m2_sr_Hz=1.0,
    )
    assert out["spectral_radiance_W_m2_sr_Hz"][0] == pytest.approx(7.0)


def test_transfer_rejects_negative_absorption():
    with pytest.raises(ValueError, match="non-negative"):
        solve_ece_radiative_transfer(
            [10e9], [0.0, 1.0], [-1.0],
            emissivity_W_m3_sr_Hz=[1.0],
        )
