import numpy as np
from tis.interferometry import analyze_interferometry


def test_interferometry_density_conversion():
    r = analyze_interferometry([0.1, 0.2], frequency_hz=100e9, sign_convention=1, path_length_m=1.0)
    assert np.all(r['line_integrated_electron_density_m-2'] > 0)
    np.testing.assert_allclose(r['path_average_electron_density_m-3'], r['line_integrated_electron_density_m-2'])
