import numpy as np
from tis.raytrace import trace_o_mode_ray


def test_uniform_density_ray_has_no_deflection():
    x = np.linspace(-0.1, 0.1, 51)
    ne = np.full_like(x, 1e17)
    result = trace_o_mode_ray(x, ne, frequency_hz=60e9, propagation_length_m=1.0, initial_x_m=0.0)
    assert abs(result['angular_deflection_rad']) < 1e-8
    assert np.max(np.abs(result['ray_position_m'])) < 1e-8
