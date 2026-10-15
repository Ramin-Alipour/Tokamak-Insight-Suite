import numpy as np
from tis.thermal_response import analyze_thermal_response


def test_thermal_response_diffusivity_and_heat_flux():
    t = np.linspace(0, 1, 101)
    temp = 300 + 10*t
    r = analyze_thermal_response(t, temp, conductivity_W_mK=10, density_kg_m3=1000, specific_heat_J_kgK=500)
    assert r['thermal_diffusivity_m2_s'] == 2e-5
    assert r['surface_heat_flux_estimate_W_m2'][-1] > 0
