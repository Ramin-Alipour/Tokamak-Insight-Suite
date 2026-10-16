from __future__ import annotations
import numpy as np
from .core import polynomial_calibration


def analyze_thermocouple(voltage_mV, *, temperature_calibration_coefficients, cold_junction_emf_mV: float = 0.0, time_s=None) -> dict:
    """Convert thermocouple EMF to temperature using a supplied polynomial calibration."""
    v = np.asarray(voltage_mV, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)) or not np.isfinite(cold_junction_emf_mV):
        raise ValueError('Voltage and cold-junction EMF must be finite and nonempty.')
    corrected_emf = v + cold_junction_emf_mV
    temperature_c = polynomial_calibration(corrected_emf, list(temperature_calibration_coefficients))
    result = {'diagnostic': 'Thermocouples / embedded sensors', 'measured_emf_mV': v,
        'cold_junction_emf_correction_mV': float(cold_junction_emf_mV),
        'temperature_C': temperature_c, 'temperature_K': temperature_c+273.15,
        'assumptions_and_limits': ['Polynomial coefficients and cold-junction compensation must match the thermocouple type and calibration standard.',
            'Lead-wire effects, sensor contact, thermal lag and radiation errors are not automatically corrected.']}
    if time_s is not None:
        t = np.asarray(time_s, dtype=float).ravel()
        tc = np.asarray(temperature_c, dtype=float).ravel()
        if tc.size != t.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError('time_s must be finite, strictly increasing and match the voltage trace.')
        result['time_s'] = t
        result['heating_rate_C_s'] = np.gradient(tc, t)
    return result
