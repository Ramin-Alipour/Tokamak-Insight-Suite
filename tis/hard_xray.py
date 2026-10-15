from __future__ import annotations
from typing import Any
import numpy as np
from .core import xray_spectrum_summary


def hard_xray_spectrum_workflow(energy_keV: Any, counts: Any, *, background_counts: Any | None = None, exposure_s: float | None = None, assume_poisson_counts: bool = False, bin_representation: str = 'counts_per_bin') -> dict[str, Any]:
    """Summarize a hard-X-ray pulse-height spectrum without response unfolding."""
    energy = np.asarray(energy_keV, dtype=float).ravel()
    signal = np.asarray(counts, dtype=float).ravel()
    if len(energy) != len(signal) or len(energy) < 2:
        raise ValueError('Energy and count arrays must have equal lengths and at least two bins.')
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(signal)):
        raise ValueError('Energy and counts must be finite.')
    if np.any(energy <= 0) or np.any(signal < 0):
        raise ValueError('Energy values must be positive and gross counts nonnegative.')
    if np.any(np.diff(energy) <= 0):
        raise ValueError('Energy bins must be strictly increasing.')
    if background_counts is not None:
        bg = np.asarray(background_counts, dtype=float).ravel()
        if len(bg) != len(signal) or not np.all(np.isfinite(bg)) or np.any(bg < 0):
            raise ValueError('Background counts must be finite, nonnegative and match the spectrum.')
    else:
        bg = None
    if exposure_s is not None and (not np.isfinite(exposure_s) or exposure_s <= 0):
        raise ValueError('exposure_s must be positive and finite.')
    summary = xray_spectrum_summary(energy, signal, background=bg, assume_poisson_counts=assume_poisson_counts, bin_representation=bin_representation)
    net = signal if bg is None else signal - bg
    summary.update({
        'gross_counts_sum': float(np.sum(signal)),
        'background_counts_sum': None if bg is None else float(np.sum(bg)),
        'net_counts_sum': float(np.sum(net)),
        'exposure_s': exposure_s,
        'gross_count_rate_s-1': None if exposure_s is None else float(np.sum(signal) / exposure_s),
        'net_count_rate_s-1': None if exposure_s is None else float(np.sum(net) / exposure_s),
        'interpretation': 'Pulse-height/count summary only. Absolute photon flux or runaway-electron energy requires detector energy/efficiency response, dead-time and pile-up corrections, and an appropriate transport model.'
    })
    return summary
