import numpy as np
from tis.particle_transport import analyze_particle_transport


def test_particle_transport_recovers_diffusivity():
    r = np.linspace(0, 1, 101)
    n = 2e19 - 1e19*r
    flux = np.full_like(r, 1e19)
    result = analyze_particle_transport(r, n, flux)
    np.testing.assert_allclose(result['effective_diffusivity_m2_s-1'], 1.0, rtol=1e-8)
