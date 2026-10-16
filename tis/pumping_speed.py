from __future__ import annotations
import numpy as np


def analyze_effective_pumping_speed(throughput_Pa_m3_s, pressure_Pa, *, pump_speed_m3_s=None, conductance_m3_s=None) -> dict:
    """Estimate effective pumping speed from Q/p and optionally apply series conductance."""
    q, p = np.broadcast_arrays(np.asarray(throughput_Pa_m3_s, dtype=float), np.asarray(pressure_Pa, dtype=float))
    if q.size == 0 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)) or np.any(q < 0) or np.any(p <= 0):
        raise ValueError('Throughput must be finite/nonnegative and pressure finite/positive.')
    local_speed = q/p
    effective = local_speed
    if (pump_speed_m3_s is None) != (conductance_m3_s is None):
        raise ValueError('pump_speed_m3_s and conductance_m3_s must be supplied together.')
    if pump_speed_m3_s is not None:
        if not np.isfinite(pump_speed_m3_s) or pump_speed_m3_s <= 0 or not np.isfinite(conductance_m3_s) or conductance_m3_s <= 0:
            raise ValueError('Pump speed and conductance must be positive and finite.')
        effective = 1.0/(1.0/pump_speed_m3_s + 1.0/conductance_m3_s)
    return {'diagnostic': 'Effective pumping speed', 'throughput_Pa_m3_s': q,
        'pressure_Pa': p, 'throughput_over_pressure_speed_m3_s': local_speed,
        'effective_speed_with_series_conductance_m3_s': effective,
        'pump_speed_m3_s': pump_speed_m3_s, 'conductance_m3_s': conductance_m3_s,
        'assumptions_and_limits': ['Throughput and pressure must use consistent SI units and refer to a defined control volume.',
            'Series conductance correction assumes a simple pump-and-duct connection.',
            'Gas dependence, molecular-flow regime and gauge-location corrections are not inferred.']}
