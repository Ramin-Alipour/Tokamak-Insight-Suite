from __future__ import annotations
import numpy as np


def analyze_rga_spectrum(mass_to_charge, ion_current_A, *, sensitivity_Pa_per_A, background_current_A=None) -> dict:
    """Convert residual-gas-analyzer ion currents to partial-pressure proxies."""
    mass = np.asarray(mass_to_charge, dtype=float).ravel()
    current = np.asarray(ion_current_A, dtype=float).ravel()
    if mass.size != current.size or mass.size < 2 or not np.all(np.isfinite(mass)) or not np.all(np.isfinite(current)):
        raise ValueError('Mass/charge and current arrays must be finite, equal length and have at least two points.')
    if np.any(np.diff(mass) <= 0):
        raise ValueError('Mass/charge values must increase strictly.')
    if background_current_A is not None:
        background = np.asarray(background_current_A, dtype=float)
        if background.ndim == 0:
            background = np.full(current.size, float(background))
        else:
            background = background.ravel()
        if background.size != current.size or not np.all(np.isfinite(background)):
            raise ValueError('Background current must be finite and scalar or match the spectrum.')
        corrected = np.clip(current-background, 0, None)
    else:
        corrected = np.clip(current, 0, None)
    sensitivity = np.asarray(sensitivity_Pa_per_A, dtype=float)
    if sensitivity.ndim == 0:
        sensitivity = np.full(current.size, float(sensitivity))
    else:
        sensitivity = sensitivity.ravel()
    if sensitivity.size != current.size or not np.all(np.isfinite(sensitivity)) or np.any(sensitivity <= 0):
        raise ValueError('Sensitivity must be positive, finite and scalar or match the spectrum.')
    pressure = corrected*sensitivity
    total = float(np.sum(pressure))
    fractions = pressure/total if total > 0 else np.zeros_like(pressure)
    return {'diagnostic': 'Residual gas analyzer (RGA)', 'mass_to_charge': mass,
        'background_corrected_ion_current_A': corrected, 'partial_pressure_proxy_Pa': pressure,
        'total_pressure_proxy_Pa': total, 'partial_pressure_fraction': fractions,
        'dominant_mass_to_charge': float(mass[int(np.argmax(pressure))]),
        'assumptions_and_limits': ['Sensitivity factors must match the analyzer, gas species, electron energy and operating conditions.',
            'Fragmentation patterns can cause one gas species to contribute to multiple mass peaks.',
            'Partial pressures are proxies unless species-specific calibration and background are established.']}
