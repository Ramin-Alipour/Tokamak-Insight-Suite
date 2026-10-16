from __future__ import annotations
import numpy as np
from .core import electron_cyclotron_resonance_field


def analyze_ecrh_power(launched_power_W, *, transmission_fraction: float = 1.0, reflection_fraction: float = 0.0, absorption_fraction: float = 1.0, frequency_hz=None, harmonic: int = 1) -> dict:
    """Estimate ECRH coupled/absorbed power and optional resonance field."""
    power = np.asarray(launched_power_W, dtype=float)
    if power.size == 0 or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError('launched_power_W must be finite and nonnegative.')
    for name, value in [('transmission_fraction', transmission_fraction), ('reflection_fraction', reflection_fraction), ('absorption_fraction', absorption_fraction)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between zero and one.')
    if transmission_fraction + reflection_fraction > 1 + 1e-12:
        raise ValueError('Transmission plus reflection fractions cannot exceed one.')
    coupled = power*transmission_fraction*(1-reflection_fraction)
    absorbed = coupled*absorption_fraction
    resonance = electron_cyclotron_resonance_field(frequency_hz, harmonic) if frequency_hz is not None else None
    return {'diagnostic': 'ECRH power', 'launched_power_W': power,
        'coupled_power_W': coupled, 'absorbed_power_W': absorbed,
        'total_absorbed_power_W': float(np.sum(absorbed)),
        'transmission_fraction': float(transmission_fraction), 'reflection_fraction': float(reflection_fraction),
        'absorption_fraction': float(absorption_fraction), 'resonance': resonance,
        'assumptions_and_limits': ['Power fractions must be measured or modeled for the relevant launcher and plasma conditions.',
            'Resonance field is a cold single-particle estimate and does not determine deposition location by itself.',
            'Relativistic effects, polarization, refraction and absorption profiles are not modeled.']}
