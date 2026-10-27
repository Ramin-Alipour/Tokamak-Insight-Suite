from __future__ import annotations
from typing import Any
import numpy as np
from .core import integrate_time_series


def magnetic_flux_from_voltage(
    time_s: Any, coil_voltage_V: Any, *, polarity_sign: int = -1,
    initial_flux_Wb: float = 0.0,
    baseline_start_s: float | None = None, baseline_end_s: float | None = None,
    drift_start_s: float | None = None, drift_end_s: float | None = None,
) -> dict[str, Any]:
    """Integrate calibrated loop voltage using Faraday's law.

    Inputs are one-dimensional, equally sized samples: time in seconds and
    calibrated loop voltage in volts. The convention is dPhi/dt = -V_loop;
    polarity_sign applies the documented sensor orientation. Optional baseline
    and drift windows must represent known zero-voltage intervals. This routine
    reconstructs flux from a loop signal; it does not solve Grad-Shafranov or
    infer a spatial poloidal-flux map from insufficient measurements.
    """
    if isinstance(polarity_sign, (bool, np.bool_)) or not isinstance(
        polarity_sign, (int, np.integer)
    ) or polarity_sign not in (-1, 1):
        raise ValueError("polarity_sign must be integer -1 or +1.")
    if isinstance(initial_flux_Wb, (bool, np.bool_)) or not np.isscalar(initial_flux_Wb) or not np.isfinite(initial_flux_Wb):
        raise ValueError("initial_flux_Wb must be a finite real scalar.")
    try:
        t = np.asarray(time_s, dtype=float)
        v = np.asarray(coil_voltage_V, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Time and voltage must be numeric one-dimensional arrays.") from exc
    if t.ndim != 1 or v.ndim != 1:
        raise ValueError("Time and voltage must be one-dimensional arrays; multidimensional inputs are not flattened.")
    if len(t) != len(v) or len(t) < 2:
        raise ValueError("Time and voltage must have equal lengths and at least two samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        raise ValueError("Time and calibrated voltage must be finite.")
    result = integrate_time_series(
        t, v, scale=float(polarity_sign), initial_value=float(initial_flux_Wb),
        baseline_start_s=baseline_start_s, baseline_end_s=baseline_end_s,
        drift_start_s=drift_start_s, drift_end_s=drift_end_s,
    )
    flux = np.asarray(result["integrated_signal"], dtype=float)
    return {
        **result, "magnetic_flux_Wb": flux,
        "flux_change_Wb": flux - float(initial_flux_Wb),
        "initial_flux_Wb": float(initial_flux_Wb), "polarity_sign": int(polarity_sign),
        "physical_quantity": "magnetic flux in Wb; flux change if initial flux is zero",
        "interpretation": (
            "Requires calibrated loop voltage and a documented sign convention. "
            "Integration does not reconstruct a full tokamak equilibrium or a spatial poloidal-flux map."
        ),
    }
