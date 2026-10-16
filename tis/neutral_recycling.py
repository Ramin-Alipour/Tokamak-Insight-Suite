from __future__ import annotations
import numpy as np

K_B = 1.380649e-23
AMU = 1.66053906660e-27


def estimate_neutral_pressure_recycling(pressure_Pa, *, gas_temperature_K: float, gas_mass_amu: float, surface_area_m2: float, incident_ion_flux_m2_s=None, recycling_coefficient: float | None = None) -> dict:
    """Estimate neutral density and thermal impingement/recycling rates from pressure."""
    p = np.asarray(pressure_Pa, dtype=float)
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('pressure_Pa must be finite and nonnegative.')
    for name, value in [('gas_temperature_K', gas_temperature_K), ('gas_mass_amu', gas_mass_amu), ('surface_area_m2', surface_area_m2)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    density = p/(K_B*gas_temperature_K)
    mean_speed = float(np.sqrt(8*K_B*gas_temperature_K/(np.pi*gas_mass_amu*AMU)))
    impingement_flux = density*mean_speed/4
    result = {'diagnostic': 'Neutral pressure / recycling estimates', 'neutral_density_m-3': density,
        'mean_thermal_speed_m_s': mean_speed, 'neutral_impingement_flux_m-2_s-1': impingement_flux,
        'surface_area_m2': float(surface_area_m2), 'impingement_rate_s-1': impingement_flux*surface_area_m2,
        'assumptions_and_limits': ['Ideal-gas and Maxwellian neutral thermal-speed approximation.',
            'Local pressure and gas temperature must represent the relevant neutral population.',
            'Impingement is not identical to net recycling or particle flux into the plasma.']}
    if (incident_ion_flux_m2_s is None) != (recycling_coefficient is None):
        raise ValueError('incident_ion_flux_m2_s and recycling_coefficient must be supplied together.')
    if incident_ion_flux_m2_s is not None:
        flux = np.asarray(incident_ion_flux_m2_s, dtype=float)
        if not np.all(np.isfinite(flux)) or np.any(flux < 0):
            raise ValueError('incident_ion_flux_m2_s must be finite and nonnegative.')
        if not np.isfinite(recycling_coefficient) or not 0 <= recycling_coefficient <= 1:
            raise ValueError('recycling_coefficient must be between zero and one.')
        recycled_flux = recycling_coefficient*flux
        result['recycled_neutral_flux_m-2_s-1'] = recycled_flux
        result['recycled_neutral_rate_s-1'] = recycled_flux*surface_area_m2
    return result
