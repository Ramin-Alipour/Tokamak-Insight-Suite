import numpy as np
import pytest
from tis.polarimetry import analyze_faraday_rotation


def test_faraday_forward_and_inverse():
    wavelength = 1e-6
    path = np.linspace(0, 1, 101)
    ne = np.full_like(path, 1e19)
    b = np.full_like(path, 1.5)
    forward = analyze_faraday_rotation(wavelength, path_m=path, electron_density_m3=ne, magnetic_field_parallel_T=b)
    result = analyze_faraday_rotation(wavelength, rotation_angle_rad=forward['predicted_rotation_angle_rad'], path_m=path, electron_density_m3=ne, magnetic_field_parallel_T=b)
    assert result['density_weighted_mean_Bparallel_T'] == pytest.approx(1.5, rel=1e-10)


def test_faraday_rejects_incomplete_profile_inputs():
    with pytest.raises(ValueError, match='must be supplied together'):
        analyze_faraday_rotation(1e-6, path_m=[0, 1])
