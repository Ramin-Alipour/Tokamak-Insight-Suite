from __future__ import annotations
import numpy as np


def analyze_lower_hybrid_power(launched_power_W, *, frequency_hz: float, transmission_fraction: float = 1.0, reflection_fraction: float = 0.0, absorption_fraction: float = 1.0, driven_current_A=None) -> dict:
    """Account for lower-hybrid launched/coupled/absorbed power and optional current-drive efficiency."""
    power = np.asarray(launched_power_W, dtype=float)
    if power.size == 0 or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError('launched_power_W must be finite and nonnegative.')
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('frequency_hz must be positive and finite.')
    for name, value in [('transmission_fraction', transmission_fraction), ('reflection_fraction', reflection_fraction), ('absorption_fraction', absorption_fraction)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between zero and one.')
    if transmission_fraction + reflection_fraction > 1 + 1e-12:
        raise ValueError('Transmission plus reflection fractions cannot exceed one.')
    coupled = power*transmission_fraction*(1-reflection_fraction)
    absorbed = coupled*absorption_fraction
    current = None if driven_current_A is None else np.asarray(driven_current_A, dtype=float)
    efficiency = None
    if current is not None:
        if not np.all(np.isfinite(current)):
            raise ValueError('driven_current_A must be finite.')
        if np.any(absorbed <= 0):
            raise ValueError('Current-drive efficiency requires positive absorbed power.')
        efficiency = current/absorbed
    return {'diagnostic': 'Lower hybrid heating/current drive', 'frequency_Hz': float(frequency_hz),
        'launched_power_W': power, 'coupled_power_W': coupled, 'absorbed_power_W': absorbed,
        'total_absorbed_power_W': float(np.sum(absorbed)), 'driven_current_A': current,
        'current_drive_efficiency_A_W': efficiency,
        'assumptions_and_limits': ['Power fractions must be supplied from launcher and plasma-coupling measurements/modeling.',
            'Current-drive efficiency is meaningful only when driven current and absorbed power use consistent definitions.',
            'Wave spectrum, accessibility, damping and current-profile deposition are not modeled.']}
