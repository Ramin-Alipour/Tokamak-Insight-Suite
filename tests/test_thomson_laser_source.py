import numpy as np
import pytest

from tokamak_analysis.thomson_inference import _laser_source_response


def test_measured_laser_profile_operator_conserves_interior_spectral_power():
    wavelength = np.linspace(520.0, 544.0, 801)
    laser_axis = np.linspace(531.7, 532.3, 301)
    laser_profile = np.exp(-0.5 * ((laser_axis - 532.0) / 0.055) ** 2)

    operator, metadata = _laser_source_response(
        wavelength, 532.0, laser_axis, laser_profile
    )

    assert operator.shape == (wavelength.size, wavelength.size)
    assert np.all(np.isfinite(operator))
    assert np.all(operator >= 0.0)
    column_sums = operator.sum(axis=0)
    assert np.all(column_sums <= 1.0 + 1e-12)
    assert np.allclose(column_sums[20:-20], 1.0, atol=1e-10)
    assert metadata["rms_width_nm"] > 0.0
    assert metadata["source_profile_samples"] == laser_axis.size


def test_measured_laser_profile_rejects_negative_density():
    wavelength = np.linspace(520.0, 544.0, 101)
    laser_axis = np.linspace(531.9, 532.1, 11)
    laser_profile = np.ones_like(laser_axis)
    laser_profile[5] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        _laser_source_response(
            wavelength, 532.0, laser_axis, laser_profile
        )
