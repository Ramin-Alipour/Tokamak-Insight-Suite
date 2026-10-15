import numpy as np
from tis.neutral_recycling import estimate_neutral_pressure_recycling


def test_neutral_pressure_and_recycling_estimates():
    r = estimate_neutral_pressure_recycling([0.1, 0.2], gas_temperature_K=300, gas_mass_amu=2, surface_area_m2=0.5, incident_ion_flux_m2_s=1e20, recycling_coefficient=0.9)
    assert np.all(r['neutral_density_m-3'] > 0)
    assert r['recycled_neutral_rate_s-1'] == 4.5e19
