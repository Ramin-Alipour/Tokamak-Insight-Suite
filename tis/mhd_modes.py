from __future__ import annotations
from typing import Any
import numpy as np
from .core import mode_number_from_channels


def mhd_mode_workflow(sensor_angles_rad: Any, synchronized_signals: Any, *, sample_rate_Hz: float, frequency_range_Hz: tuple[float, float] | None = None, minimum_coherence: float = 0.2, convention: str = 'wave_cos_wt_minus_n_theta') -> dict[str, Any]:
    """Estimate a coherent mode frequency and spatial mode number from a sensor ring."""
    theta = np.asarray(sensor_angles_rad, dtype=float).ravel()
    data = np.asarray(synchronized_signals, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(theta):
        raise ValueError('Signals must have shape (number of sensor angles, time samples).')
    if len(np.unique(np.round(np.mod(theta, 2*np.pi), 12))) != len(theta):
        raise ValueError('Sensor angles must be unique modulo 2π.')
    result = mode_number_from_channels(theta, data, float(sample_rate_Hz), frequency_range_hz, float(minimum_coherence), convention)
    result['sensor_angles_rad'] = theta.tolist()
    result['sample_rate_Hz'] = float(sample_rate_Hz)
    result['interpretation'] = 'Mode number is constrained by sensor geometry, phase convention, coherence and spatial aliasing; it is not a universal mode identification.'
    return result
