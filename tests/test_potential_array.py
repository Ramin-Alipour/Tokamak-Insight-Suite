import numpy as np
import pytest
from tis.potential_array import analyze_potential_array


def test_potential_array_recovers_linear_electric_field():
    x = np.array([0.0, 0.01, 0.02, 0.03])
    v = 5.0 - 20.0*x
    result = analyze_potential_array(x, v)
    np.testing.assert_allclose(result['electric_field_V_m'], 20.0, atol=1e-10)


def test_potential_array_time_series_shape_and_time_check():
    x = np.array([0.0, 0.01, 0.02])
    v = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)
    result = analyze_potential_array(x, v, time_s=[0, 1, 2])
    assert result['potential_fluctuation_rms_by_channel_V'].shape == (3,)
    with pytest.raises(ValueError, match='time_s'):
        analyze_potential_array(x, v, time_s=[0, 1])
