"""Tests for electric-field extraction from probe potential pairs."""
import numpy as np
import pytest

from tis.core import electric_field_from_potential_difference


def test_electric_field_uses_explicit_gradient_sign_and_spacing():
    va = np.array([10.0, 12.0, 14.0])
    vb = np.array([8.0, 12.0, 18.0])
    np.testing.assert_allclose(electric_field_from_potential_difference(va, vb, 0.02), [100.0, 0.0, -200.0])
    np.testing.assert_allclose(electric_field_from_potential_difference(va, vb, 0.02, field_sign=1), [-100.0, 0.0, 200.0])


def test_electric_field_preserves_missing_values():
    out = electric_field_from_potential_difference([1.0, np.nan], [2.0, 3.0], 0.5)
    assert out[0] == pytest.approx(-2.0)
    assert np.isnan(out[1])


@pytest.mark.parametrize("spacing", [0.0, np.nan, np.inf])
def test_electric_field_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="separation"):
        electric_field_from_potential_difference([1.0, 2.0], [2.0, 3.0], spacing)
