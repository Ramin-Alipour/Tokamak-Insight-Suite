from __future__ import annotations
import numpy as np


def analyze_electron_ion_heating_power(absorbed_power_W, *, electron_fraction: float, ion_fraction: float | None = None, electron_power_W=None, ion_power_W=None) -> dict:
    """Partition absorbed heating power into electron and ion channels."""
    power = np.asarray(absorbed_power_W, dtype=float)
    if power.size == 0 or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError('absorbed_power_W must be finite and nonnegative.')
    if not np.isfinite(electron_fraction) or not 0 <= electron_fraction <= 1:
        raise ValueError('electron_fraction must be between zero and one.')
    if ion_fraction is not None and (not np.isfinite(ion_fraction) or not 0 <= ion_fraction <= 1):
        raise ValueError('ion_fraction must be between zero and one.')
    if ion_fraction is not None and abs(electron_fraction+ion_fraction-1) > 1e-6:
        raise ValueError('Electron and ion fractions must sum to one.')
    pe = np.asarray(electron_power_W, dtype=float) if electron_power_W is not None else power*electron_fraction
    pi = np.asarray(ion_power_W, dtype=float) if ion_power_W is not None else power*(ion_fraction if ion_fraction is not None else 1-electron_fraction)
    if not np.all(np.isfinite(pe)) or not np.all(np.isfinite(pi)) or np.any(pe < 0) or np.any(pi < 0):
        raise ValueError('Electron and ion powers must be finite and nonnegative.')
    if pe.shape != power.shape or pi.shape != power.shape:
        raise ValueError('Explicit electron/ion power arrays must match absorbed_power_W shape.')
    residual = power-pe-pi
    return {'diagnostic': 'Electron/ion heating power', 'absorbed_power_W': power,
        'electron_heating_power_W': pe, 'ion_heating_power_W': pi,
        'power_partition_residual_W': residual,
        'electron_fraction_actual': np.divide(pe, power, out=np.full_like(power, np.nan), where=power>0),
        'ion_fraction_actual': np.divide(pi, power, out=np.full_like(power, np.nan), where=power>0),
        'assumptions_and_limits': ['Heating fractions must come from measurements or a suitable deposition model.',
            'Power partition does not by itself determine spatial deposition or thermal equilibration.',
            'A nonzero residual indicates unassigned power or inconsistent explicit electron/ion inputs.']}
