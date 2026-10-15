from __future__ import annotations
import numpy as np

E_CHARGE = 1.602176634e-19


def analyze_nbi_power(beam_energy_keV, particle_rate_s, *, neutralization_fraction: float = 1.0, coupling_fraction: float = 1.0) -> dict:
    """Estimate injected NBI power from beam energy and particle rate."""
    energy = np.asarray(beam_energy_keV, dtype=float)
    rate = np.asarray(particle_rate_s, dtype=float)
    if energy.size == 0 or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(rate)) or np.any(energy < 0) or np.any(rate < 0):
        raise ValueError('Beam energy and particle rate must be finite and nonnegative.')
    try:
        energy, rate = np.broadcast_arrays(energy, rate)
    except ValueError as exc:
        raise ValueError('Beam energy and particle rate must be broadcast-compatible.') from exc
    for name, value in [('neutralization_fraction', neutralization_fraction), ('coupling_fraction', coupling_fraction)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between zero and one.')
    power = energy*1e3*E_CHARGE*rate*neutralization_fraction*coupling_fraction
    return {'diagnostic': 'Neutral beam injection (NBI) power', 'beam_energy_keV': energy,
        'particle_rate_s-1': rate, 'injected_power_W': power,
        'total_injected_power_W': float(np.sum(power)),
        'neutralization_fraction': float(neutralization_fraction), 'coupling_fraction': float(coupling_fraction),
        'assumptions_and_limits': ['Particle rate must represent the beam species at the defined injection point.',
            'Neutralization and coupling fractions must correspond to the beamline and plasma coupling convention.',
            'This estimate does not include shine-through, orbit losses or beam deposition profiles.']}
