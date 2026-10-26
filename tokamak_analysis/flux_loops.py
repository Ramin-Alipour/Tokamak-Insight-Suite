"""Flux-loop signal analysis using Faraday's law.

Voltage convention: V_loop = -N * d(Phi)/dt.  Consequently the
reconstructed flux is Phi(t) = Phi0 - integral(V_loop/N dt).
Inputs must be calibrated loop voltage in volts and time in seconds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid


@dataclass(frozen=True)
class FluxLoopResult:
    time_s: np.ndarray
    voltage_V: np.ndarray
    flux_change_Wb: np.ndarray
    flux_Wb: np.ndarray
    loop_voltage_integral_Vs: np.ndarray
    turns: int
    initial_flux_Wb: float
    voltage_offset_V: float
    sign_convention: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly arrays and scalar metadata."""
        result = asdict(self)
        for key in ("time_s", "voltage_V", "flux_change_Wb", "flux_Wb",
                    "loop_voltage_integral_Vs"):
            result[key] = result[key].tolist()
        return result


def analyze_flux_loop(
    time_s,
    voltage_V,
    *,
    turns: int = 1,
    initial_flux_Wb: float = 0.0,
    voltage_offset_V: float = 0.0,
    polarity: int = 1,
    max_gap_s: float | None = None,
) -> FluxLoopResult:
    """Reconstruct magnetic flux from a calibrated flux-loop voltage.

    Args:
        time_s: strictly increasing sample times in seconds.
        voltage_V: calibrated loop voltage samples in volts.
        turns: positive integer number of turns in the loop.
        initial_flux_Wb: integration constant in webers.
        voltage_offset_V: constant voltage baseline to subtract, in volts.
        polarity: sensor wiring convention, either +1 or -1.
        max_gap_s: optional maximum allowed adjacent sample gap in seconds.

    Returns:
        FluxLoopResult, including flux change relative to the first sample,
        absolute flux using initial_flux_Wb, and quality diagnostics.

    Raises:
        ValueError: for invalid shapes, non-finite values, invalid calibration,
        non-monotonic time, or excessive sampling gaps.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(voltage_V, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or t.size != v.size or t.size < 2:
        raise ValueError("time_s and voltage_V must be equal-length 1-D arrays with at least two samples")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        raise ValueError("time_s and voltage_V must contain only finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if isinstance(turns, bool) or not isinstance(turns, (int, np.integer)) or turns < 1:
        raise ValueError("turns must be a positive integer")
    if polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    if not np.isfinite(initial_flux_Wb) or not np.isfinite(voltage_offset_V):
        raise ValueError("initial_flux_Wb and voltage_offset_V must be finite")
    dt = np.diff(t)
    if max_gap_s is not None:
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")
        if np.any(dt > max_gap_s):
            raise ValueError("time series contains gaps larger than max_gap_s")

    corrected = polarity * (v - float(voltage_offset_V))
    integral = cumulative_trapezoid(corrected, t, initial=0.0)
    flux_change = -integral / int(turns)
    flux = float(initial_flux_Wb) + flux_change
    diagnostics = {
        "sample_count": int(t.size),
        "duration_s": float(t[-1] - t[0]),
        "median_sample_interval_s": float(np.median(dt)),
        "minimum_sample_interval_s": float(np.min(dt)),
        "maximum_sample_interval_s": float(np.max(dt)),
        "maximum_absolute_voltage_V": float(np.max(np.abs(corrected))),
        "voltage_mean_V": float(np.mean(corrected)),
        "voltage_rms_V": float(np.sqrt(np.mean(corrected**2))),
        "flux_change_peak_to_peak_Wb": float(np.ptp(flux_change)),
        "baseline_corrected": bool(voltage_offset_V != 0.0),
        "absolute_flux_depends_on_initial_condition": True,
    }
    return FluxLoopResult(
        time_s=t.copy(), voltage_V=corrected.copy(),
        flux_change_Wb=flux_change, flux_Wb=flux,
        loop_voltage_integral_Vs=integral, turns=int(turns),
        initial_flux_Wb=float(initial_flux_Wb),
        voltage_offset_V=float(voltage_offset_V),
        sign_convention="V_loop = -N*dPhi/dt; polarity applied to (V-offset)",
        diagnostics=diagnostics,
    )


def flux_from_voltage(time_s, voltage_V, **kwargs) -> FluxLoopResult:
    """Compatibility alias for :func:`analyze_flux_loop`."""
    return analyze_flux_loop(time_s, voltage_V, **kwargs)
