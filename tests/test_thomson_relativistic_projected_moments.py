import numpy as np
import pytest

from tis.products.thomson.physics.thomson_distribution import (
    _projected_evdf_relativistic_lab_moments,
)


def _weights(v):
    dv = np.diff(v)
    w = np.empty_like(v)
    w[0] = 0.5 * dv[0]
    w[-1] = 0.5 * dv[-1]
    w[1:-1] = 0.5 * (dv[:-1] + dv[1:])
    return w


def test_relativistic_projected_energy_flux_uses_exact_lorentz_energy():
    c = 299792458.0
    me = 9.1093837139e-31
    v = np.linspace(-0.35 * c, 0.35 * c, 801)
    # A deliberately asymmetric two-population projected marginal.
    f = (
        np.exp(-0.5 * ((v - 0.12 * c) / (0.045 * c)) ** 2)
        + 0.35 * np.exp(-0.5 * ((v + 0.08 * c) / (0.025 * c)) ** 2)
    )
    result = _projected_evdf_relativistic_lab_moments(v, f)
    p = _weights(v) * f
    p /= p.sum()
    gamma = 1.0 / np.sqrt(1.0 - (v / c) ** 2)
    energy = (gamma - 1.0) * me * c**2
    expected_energy = float(p @ energy)
    expected_flux = float(p @ (energy * v))
    assert result["mean_kinetic_energy_eV"] == pytest.approx(
        expected_energy / 1.602176634e-19, rel=2e-12
    )
    assert result["kinetic_energy_flux_per_density_W_m"] == pytest.approx(
        expected_flux, rel=2e-12
    )
    assert result["heat_flux_status"] == "not_inferred_from_single_projected_marginal"


def test_relativistic_projected_moments_covariance_matches_finite_difference():
    c = 299792458.0
    v = np.linspace(-0.2 * c, 0.2 * c, 121)
    f = np.exp(-0.5 * ((v - 0.035 * c) / (0.055 * c)) ** 2)
    scale = np.max(f)
    f = f / scale
    covariance = np.diag((0.01 * f + 1e-5) ** 2)
    result = _projected_evdf_relativistic_lab_moments(v, f, covariance)
    w = _weights(v)
    c0 = 299792458.0
    me = 9.1093837139e-31

    def observables(density):
        probability = w * density
        probability = probability / probability.sum()
        gamma = 1.0 / np.sqrt(1.0 - (v / c0) ** 2)
        energy = (gamma - 1.0) * me * c0**2
        return np.array([
            probability @ v,
            probability @ energy,
            probability @ (energy * v),
        ])

    jacobian = np.empty((3, v.size))
    for j in range(v.size):
        step = 1e-6 * max(abs(f[j]), 1e-3)
        plus, minus = f.copy(), f.copy()
        plus[j] += step
        minus[j] -= step
        jacobian[:, j] = (observables(plus) - observables(minus)) / (2.0 * step)
    expected = jacobian @ covariance @ jacobian.T
    np.testing.assert_allclose(result["moment_covariance"], expected, rtol=2e-6, atol=1e-25)


def test_relativistic_projected_moments_reject_superluminal_grid():
    v = np.linspace(-1.01, 1.01, 21) * 299792458.0
    f = np.ones_like(v)
    with pytest.raises(ValueError, match=r"\|v\| < c"):
        _projected_evdf_relativistic_lab_moments(v, f)
