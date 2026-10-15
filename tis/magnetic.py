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



def analyze_toroidal_field(
    signal,
    *,
    calibration_T_per_unit: float,
    offset_T: float = 0.0,
    signal_uncertainty: float | None = None,
    calibration_relative_uncertainty: float | None = None,
    reference_radius_m: float | None = None,
    evaluation_radius_m=None,
) -> dict:
    """Calibrate a toroidal-field signal and optionally apply approximate 1/R scaling.

    Calibration must come from the actual sensor/coil geometry and electronics.
    The optional 1/R scaling is a large-aspect-ratio vacuum-field approximation.
    """
    x = np.asarray(signal, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("signal must contain finite values.")
    if not np.isfinite(calibration_T_per_unit) or calibration_T_per_unit == 0:
        raise ValueError("calibration_T_per_unit must be finite and nonzero.")
    if not np.isfinite(offset_T):
        raise ValueError("offset_T must be finite.")
    for name, value in (("signal_uncertainty", signal_uncertainty),
                        ("calibration_relative_uncertainty", calibration_relative_uncertainty)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")
    field = x * calibration_T_per_unit + offset_T
    field_uncertainty = None
    if signal_uncertainty is not None or calibration_relative_uncertainty is not None:
        variance = np.zeros_like(field, dtype=float)
        if signal_uncertainty is not None:
            variance += (calibration_T_per_unit * signal_uncertainty) ** 2
        if calibration_relative_uncertainty is not None:
            variance += (x * calibration_T_per_unit * calibration_relative_uncertainty) ** 2
        field_uncertainty = np.sqrt(variance)
    radius_field = None
    if evaluation_radius_m is not None or reference_radius_m is not None:
        if reference_radius_m is None or evaluation_radius_m is None:
            raise ValueError("reference_radius_m and evaluation_radius_m must be supplied together.")
        if not np.isfinite(reference_radius_m) or reference_radius_m <= 0:
            raise ValueError("reference_radius_m must be positive and finite.")
        radius = np.asarray(evaluation_radius_m, dtype=float)
        if radius.size == 0 or not np.all(np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("evaluation_radius_m must contain positive finite radii.")
        radius_field = field * reference_radius_m / radius
    return {
        "diagnostic": "Toroidal-field coils / magnetic probes",
        "toroidal_field_T": field,
        "toroidal_field_standard_uncertainty_T": field_uncertainty,
        "field_at_evaluation_radius_T": radius_field,
        "calibration_T_per_unit": float(calibration_T_per_unit),
        "offset_T": float(offset_T),
        "assumptions_and_limits": [
            "Calibration coefficient and offset must be determined from the actual sensor, coil geometry and electronics.",
            "Uncertainty propagation includes only explicitly supplied independent signal and relative calibration uncertainties.",
            "Optional 1/R scaling assumes a vacuum-like toroidal field and does not represent full equilibrium geometry.",
        ],
    }
