from __future__ import annotations
import numpy as np
from .core import reflectometry_cutoff_density


def analyze_reflectometry_sweep(frequency_hz, phase_rad, *, mode='O') -> dict:
    """Convert an O-mode reflectometry frequency sweep to cutoff-density mapping and group-delay proxy."""
    frequency = np.asarray(frequency_hz, dtype=float).ravel()
    phase = np.asarray(phase_rad, dtype=float).ravel()
    if frequency.size != phase.size or frequency.size < 3:
        raise ValueError('Frequency and phase arrays must have equal lengths of at least three.')
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(phase)) or np.any(frequency <= 0):
        raise ValueError('Frequency must be positive and both arrays finite.')
    if np.any(np.diff(frequency) <= 0):
        raise ValueError('Frequency sweep must be strictly increasing.')
    mode = mode.upper()
    if mode != 'O':
        raise ValueError('Only O-mode cutoff is implemented; X-mode requires B-field and branch-specific model.')
    cutoff = np.array([reflectometry_cutoff_density(float(f), mode='O')['cutoff_density_m-3'] for f in frequency])
    unwrapped = np.unwrap(phase)
    group_delay = np.gradient(unwrapped, 2*np.pi*frequency)
    return {'diagnostic': 'Microwave reflectometry', 'frequency_Hz': frequency,
        'unwrapped_phase_rad': unwrapped, 'O_mode_cutoff_density_m-3': cutoff,
        'phase_group_delay_proxy_s': group_delay,
        'assumptions_and_limits': [
            'Cold, collisionless O-mode cutoff relation is used for the density mapping.',
            'Group-delay proxy is derived from phase slope and depends on phase convention and system delay calibration.',
            'A local density profile requires propagation/cutoff-layer inversion and equilibrium information.'
        ]}
