import numpy as np
from tis.ir_thermography import analyze_ir_thermography


def test_ir_map_outputs_finite_radiative_flux():
    data = np.full((5, 6), 500.0)
    result = analyze_ir_thermography(data, emissivity=0.8, reflected_temperature_K=300.0, pixel_size_x_m=0.001, pixel_size_y_m=0.001)
    assert result['peak_temperature_K'] > 0
    assert np.all(np.isfinite(result['radiative_heat_flux_W_m2']))
