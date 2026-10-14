"""Tests for ion sound speed and Mach-number consistency."""
import numpy as np
import pytest

from tis.core import ion_sound_speed_m_s, mach_number_from_velocity


def test_ion_sound_speed_matches_paper_equation():
    result = ion_sound_speed_m_s(14.0, 2.8, 1.0073)
    expected = np.sqrt(1.602176634e-19 * (14.0 + 2.8) / (1.0073 * 1.66053906660e-27))
    assert result["ion_sound_speed_m_s"] == pytest.approx(expected)
    assert result["ion_sound_speed_m_s"] == pytest.approx(4.01e4, rel=0.01)


def test_mach_number_from_velocity_is_signed_and_vectorized():
    cs = ion_sound_speed_m_s(14.0, 2.8, 1.0073)["ion_sound_speed_m_s"]
    mach = mach_number_from_velocity([17_000.0, -17_000.0], cs)
    np.testing.assert_allclose(mach, [17_000.0 / cs, -17_000.0 / cs])


def test_mach_number_scalar_input_returns_scalar():
    cs = ion_sound_speed_m_s(14.0, 2.8, 1.0073)["ion_sound_speed_m_s"]
    result = mach_number_from_velocity(17_000.0, cs)
    assert isinstance(result, float)
    assert result == pytest.approx(17_000.0 / cs)


def test_mach_number_rejects_invalid_sound_speed():
    with pytest.raises(ValueError, match="Sound speed"):
        mach_number_from_velocity([1.0], 0.0)
