import numpy as np
import pytest

from tokamak_analysis.cts_relativistic import _momentum_grid


def test_drifted_relativistic_maxwell_juttner_is_lorentz_boosted_and_normalized():
    mass = 9.1093837015e-31
    temperature = 2.0e5
    drift = 0.12 * 299792458.0
    distribution = {
        "kind": "drifted_relativistic_maxwell_juttner",
        "temperature_eV": temperature,
        "drift_velocity_m_s": drift,
    }
    pp, pa, f0, df_perp, df_parallel, weights = _momentum_grid(
        distribution, mass, 32
    )
    assert np.isclose(np.sum(weights * f0), 1.0, rtol=1e-10)
    mean_parallel_momentum = np.sum(weights * f0 * pa)
    assert mean_parallel_momentum > 0.0
    assert np.all(np.isfinite(df_perp))
    assert np.all(np.isfinite(df_parallel))


def test_zero_drift_limit_matches_stationary_maxwell_juttner():
    mass = 9.1093837015e-31
    stationary = _momentum_grid(
        {"kind": "relativistic_maxwell_juttner", "temperature_eV": 1.0e5},
        mass, 24,
    )
    zero_drift = _momentum_grid(
        {"kind": "drifted_relativistic_maxwell_juttner",
         "temperature_eV": 1.0e5, "drift_velocity_m_s": 0.0},
        mass, 24,
    )
    for left, right in zip(stationary, zero_drift):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=0.0)


def test_drifted_relativistic_maxwell_juttner_rejects_superluminal_drift():
    with pytest.raises(ValueError, match="strictly subluminal"):
        _momentum_grid(
            {"kind": "drifted_relativistic_maxwell_juttner",
             "temperature_eV": 1.0e5, "drift_velocity_m_s": 299792458.0},
            9.1093837015e-31, 16,
        )
