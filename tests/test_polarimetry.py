import numpy as np
import pytest
from tokamak_analysis.polarimetry import analyze_polarimetry, _FARADAY_COEFFICIENT

def test_faraday_rotation_recovers_line_integral_and_average_field():
    wavelength = np.array([10.6e-6, 10.6e-6]); ne = 2.0e19; length = 0.8; field = np.array([0.4, -0.2])
    rotation = _FARADAY_COEFFICIENT * wavelength**2 * ne * length * field
    out = analyze_polarimetry(wavelength, rotation, electron_density_m3=ne, path_length_m=length)
    np.testing.assert_allclose(out["average_B_parallel_T"], field, rtol=1e-12)
    np.testing.assert_allclose(out["line_integrated_ne_B_parallel_m-2_T"], ne * length * field, rtol=1e-12)

def test_faraday_uncertainty_propagation():
    out = analyze_polarimetry([1e-6, 2e-6], [0.01, -0.02], rotation_uncertainty_rad=1e-4, wavelength_uncertainty_m=1e-9, electron_density_m3=1e19, path_length_m=1.0, density_uncertainty_m3=1e17, path_length_uncertainty_m=0.01)
    assert np.all(np.isfinite(out["average_B_parallel_uncertainty_T"])) and np.all(out["average_B_parallel_uncertainty_T"] > 0)

@pytest.mark.parametrize("kwargs", [{"wavelength_m": [0.0], "rotation_rad": [0.1]}, {"wavelength_m": [1e-6], "rotation_rad": [float("nan")]}, {"wavelength_m": [1e-6, 2e-6], "rotation_rad": [0.1]}, {"wavelength_m": [1e-6], "rotation_rad": [0.1], "electron_density_m3": 1e19}, {"wavelength_m": [1e-6], "rotation_rad": [0.1], "rotation_uncertainty_rad": -1}])
def test_invalid_inputs_fail_explicitly(kwargs):
    with pytest.raises(ValueError): analyze_polarimetry(**kwargs)

def test_zero_rotation_returns_zero_integral():
    out = analyze_polarimetry([1e-6], [0.0]); assert out["line_integrated_ne_B_parallel_m-2_T"][0] == 0.0
