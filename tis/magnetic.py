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
    """Integrate calibrated loop voltage using V_loop = -d(Phi)/dt.

    Input voltage must already be calibrated in volts. The caller supplies the
    loop orientation via polarity_sign; no geometry or sensor transfer function
    is inferred. Baseline/drift windows must be known zero-voltage intervals.
    """
    if isinstance(polarity_sign, (bool, np.bool_)) or polarity_sign not in (-1, 1):
        raise ValueError("polarity_sign must be -1 or +1.")
    if not np.isfinite(initial_flux_Wb):
        raise ValueError("initial_flux_Wb must be finite.")
    t = np.asarray(time_s, dtype=float).ravel()
    v = np.asarray(coil_voltage_V, dtype=float).ravel()
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
            "Integration does not reconstruct a full tokamak equilibrium."
        ),
    }
