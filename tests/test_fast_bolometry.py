import numpy as np
from tis.fast_bolometry import analyze_fast_bolometry


def test_fast_bolometry_time_resolved_reconstruction():
    A = np.eye(3)
    y = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
    result = analyze_fast_bolometry([0., 1., 2.], y, A, [1., 1., 1.], regularization=0)
    assert result['emissivity_reconstruction'].shape == (3, 3)
    assert result['integrated_radiated_power_proxy_W'].shape == (3,)
