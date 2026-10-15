import numpy as np


def analyze_thermal_response(time_s, surface_temperature_K, *, conductivity_W_mK: float, density_kg_m3: float, specific_heat_J_kgK: float) -> dict:
    """Estimate semi-infinite-solid thermal response and surface heat flux from temperature history."""
    t = np.asarray(time_s, dtype=float).ravel()
    temp = np.asarray(surface_temperature_K, dtype=float).ravel()
    if t.size < 3 or t.size != temp.size or not np.all(np.isfinite(t)) or not np.all(np.isfinite(temp)) or np.any(np.diff(t) <= 0):
        raise ValueError('Time and temperature must be finite equal-length arrays with at least three samples and increasing time.')
    for name, value in [('conductivity_W_mK', conductivity_W_mK), ('density_kg_m3', density_kg_m3), ('specific_heat_J_kgK', specific_heat_J_kgK)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    alpha = conductivity_W_mK/(density_kg_m3*specific_heat_J_kgK)
    dt = np.diff(t)
    dtemp = np.diff(temp)
    mid = 0.5*(t[:-1]+t[1:])
    q = np.zeros_like(t)
    for j in range(1, t.size):
        lag = t[j]-mid[:j]
        q[j] = conductivity_W_mK/np.sqrt(np.pi*alpha)*np.sum(dtemp[:j]/np.sqrt(lag))
    return {'diagnostic': 'Thermal response / heat diffusion', 'time_s': t,
        'thermal_diffusivity_m2_s': float(alpha), 'surface_heat_flux_estimate_W_m2': q,
        'thermal_diffusion_length_m': np.sqrt(alpha*np.maximum(t-t[0], 0)),
        'assumptions_and_limits': ['Semi-infinite homogeneous solid with constant thermal properties is assumed.',
            'Surface heat-flux estimate uses a discretized semi-infinite conduction response and is sensitive to temperature noise.',
            'Contact resistance, finite geometry, radiation and latent heat are not included.']}
