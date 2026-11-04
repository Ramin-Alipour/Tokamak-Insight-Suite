import numpy as np
import pytest
from scipy.constants import Boltzmann, elementary_charge
from tokamak_analysis.ece_kinetic import integrate_kinetic_ece_radiative_transfer


def test_optically_thin_transfer_matches_formal_solution():
    te, alpha, length = 100.0, 1e-5, 2.0
    out = integrate_kinetic_ece_radiative_transfer([80e9, 100e9], [length], [alpha], [te])
    expected = te * elementary_charge / Boltzmann * (1 - np.exp(-alpha * length))
    np.testing.assert_allclose(out["brightness_temperature_out_K"], expected)
    np.testing.assert_allclose(out["total_optical_depth"], alpha * length)


def test_optically_thick_limit_reaches_source_function():
    te = 250.0
    out = integrate_kinetic_ece_radiative_transfer(
        100e9, [1.0], [100.0], [te], incident_brightness_temperature_K=10.0)
    assert out["brightness_temperature_out_K"][0] == pytest.approx(
        te * elementary_charge / Boltzmann, rel=1e-12)
    assert out["transmission"][0] == 0.0


def test_segment_order_is_respected():
    a = integrate_kinetic_ece_radiative_transfer(90e9, [1, 1], [1, 1], [100, 500])
    b = integrate_kinetic_ece_radiative_transfer(90e9, [1, 1], [1, 1], [500, 100])
    assert a["brightness_temperature_out_K"][0] != pytest.approx(b["brightness_temperature_out_K"][0])


def test_transfer_rejects_gain_and_shape_mismatch():
    with pytest.raises(ValueError, match="non-negative"):
        integrate_kinetic_ece_radiative_transfer(90e9, [1], [-1], [100])
    with pytest.raises(ValueError, match="shape"):
        integrate_kinetic_ece_radiative_transfer(90e9, [1, 2], [1], [100, 200])
