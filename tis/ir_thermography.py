from __future__ import annotations
import numpy as np

STEFAN_BOLTZMANN = 5.670374419e-8


def analyze_ir_thermography(apparent_temperature_K, *, emissivity: float, reflected_temperature_K: float, pixel_size_x_m: float, pixel_size_y_m: float, time_s=None) -> dict:
    """Apply graybody radiance correction and compute radiative heat-flux proxy and gradients."""
    temp = np.asarray(apparent_temperature_K, dtype=float)
    if temp.ndim not in (2, 3) or min(temp.shape[-2:]) < 2 or not np.all(np.isfinite(temp)) or np.any(temp <= 0):
        raise ValueError('Temperature data must be a finite positive 2-D map or time sequence of maps in kelvin.')
    for name, value in [('emissivity', emissivity), ('reflected_temperature_K', reflected_temperature_K), ('pixel_size_x_m', pixel_size_x_m), ('pixel_size_y_m', pixel_size_y_m)]:
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite.')
    if not 0 < emissivity <= 1 or reflected_temperature_K <= 0 or pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError('Emissivity must be in (0,1]; temperatures and pixel sizes must be positive.')
    rad = (temp**4 - (1-emissivity)*reflected_temperature_K**4) / emissivity
    if np.any(rad <= 0):
        raise ValueError('Graybody radiance correction is nonpositive; check temperature and emissivity inputs.')
    corrected = rad**0.25
    q_rad = emissivity * STEFAN_BOLTZMANN * (corrected**4 - reflected_temperature_K**4)
    gx = np.gradient(corrected, pixel_size_x_m, axis=-1, edge_order=1)
    gy = np.gradient(corrected, pixel_size_y_m, axis=-2, edge_order=1)
    result = {'diagnostic': 'IR thermography', 'surface_temperature_corrected_K': corrected,
        'radiative_heat_flux_W_m2': q_rad, 'temperature_gradient_x_K_m': gx, 'temperature_gradient_y_K_m': gy,
        'peak_temperature_K': float(np.max(corrected)), 'peak_radiative_heat_flux_W_m2': float(np.max(q_rad)),
        'emissivity': float(emissivity), 'reflected_temperature_K': float(reflected_temperature_K),
        'assumptions_and_limits': ['Graybody radiance correction assumes calibrated apparent temperature and a uniform emissivity.',
            'Radiative heat flux is not total plasma heat flux; conduction, convection, surface properties and view factors matter.',
            'Temperature gradients are image-plane gradients and require spatial calibration.']}
    if temp.ndim == 3:
        if time_s is None:
            raise ValueError('time_s is required for a time sequence of thermal maps.')
        t = np.asarray(time_s, dtype=float).ravel()
        if t.size != temp.shape[0] or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError('time_s must be finite, strictly increasing and match the time dimension.')
        result['time_s'] = t
        result['peak_temperature_by_time_K'] = np.max(corrected, axis=(1, 2))
        result['maximum_heating_rate_K_s'] = float(np.max(np.gradient(corrected, t, axis=0)))
    return result
