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



def test_relativistic_three_wave_vertex_scales_with_incident_field_amplitude():
    from tokamak_analysis.cts_relativistic import relativistic_gyrotropic_three_wave_vertex

    c_light = 299792458.0
    omega_i = 2.0 * np.pi * 100.0e9
    omega_f = 2.0 * np.pi * 1.0e6
    k_i = np.array([omega_i / c_light, 0.0, 0.0])
    k_f = np.array([omega_f / c_light, 0.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])
    distribution = {
        "kind": "relativistic_maxwell_juttner",
        "temperature_eV": 2.0e4,
    }
    common = dict(
        omega_incident_rad_s=omega_i,
        wavevector_incident_rad_m=k_i,
        omega_fluctuation_rad_s=omega_f,
        wavevector_fluctuation_rad_m=k_f,
        magnetic_field_T=b,
        density_m3=1.0e19,
        mass_kg=9.1093837015e-31,
        charge_C=-1.602176634e-19,
        distribution=distribution,
        quadrature_order=4,
        gyrophase_points=16,
        resonance_width_rad_s=1.0e7,
    )
    unit = relativistic_gyrotropic_three_wave_vertex(
        incident_electric_field=np.array([0.0, 1.0, 0.0]), **common
    )["current_vertex_A_m2_per_V_m"]
    doubled = relativistic_gyrotropic_three_wave_vertex(
        incident_electric_field=np.array([0.0, 2.0, 0.0]), **common
    )["current_vertex_A_m2_per_V_m"]
    np.testing.assert_allclose(doubled, 2.0 * unit, rtol=2e-10, atol=1e-30)
