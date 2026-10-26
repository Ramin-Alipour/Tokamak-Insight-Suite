"""Toroidal-field coil / magnetic-probe calibrated signal analysis."""
from __future__ import annotations
import numpy as np

def analyze_toroidal_field_coil(time_s, signal, *, calibration_T_per_unit: float,
        signal_kind: str = "field", signal_uncertainty=None,
        calibration_uncertainty_T_per_unit: float = 0.0, baseline=None,
        pickup_voltage_V=None, pickup_sensitivity_V_per_T_per_s=None,
        initial_field_T=None, pickup_voltage_baseline_V: float = 0.0,
        pickup_voltage_uncertainty_V=None,
        pickup_sensitivity_uncertainty_V_per_T_per_s: float = 0.0,
        initial_field_uncertainty_T: float = 0.0):
    """Convert calibrated coil signal samples to toroidal field and report quality metrics.

    Voltage input is converted only by the explicit T/V coefficient; no integration
    or coil transfer function is assumed. Geometry and absolute calibration are external.
    """
    t, x = np.asarray(time_s, dtype=float), np.asarray(signal, dtype=float)
    if t.ndim != 1 or x.ndim != 1 or t.size != x.size or t.size < 2:
        raise ValueError("time_s and signal must be 1-D arrays of equal length >= 2")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError("time_s and signal must contain only finite values")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")
    if signal_kind not in {"field", "current", "voltage"}:
        raise ValueError("signal_kind must be field, current, or voltage")
    k, sk = float(calibration_T_per_unit), float(calibration_uncertainty_T_per_unit)
    if not np.isfinite(k) or not np.isfinite(sk) or sk < 0:
        raise ValueError("Calibration coefficient must be finite and uncertainty non-negative")
    if signal_kind == "field" and not np.isclose(k, 1.0):
        raise ValueError("field input requires calibration_T_per_unit=1")
    if baseline is not None and not np.isfinite(float(baseline)):
        raise ValueError("baseline must be finite")
    corrected = x - (0.0 if baseline is None else float(baseline))
    bt = corrected * k
    out = {"time_s": t.copy(), "signal_corrected": corrected, "B_toroidal_T": bt,
        "signal_kind": signal_kind, "calibration_T_per_unit": k,
        "baseline_input_units": baseline, "mean_B_toroidal_T": float(np.mean(bt)),
        "rms_B_toroidal_T": float(np.sqrt(np.mean(bt**2))),
        "std_B_toroidal_T": float(np.std(bt, ddof=1)),
        "peak_to_peak_B_toroidal_T": float(np.ptp(bt)),
        "sampling_rate_Hz": float(1.0 / np.median(dt)),
        "sampling_interval_jitter_fraction": float(np.std(dt) / np.mean(dt)),
        "physical_relation": "B_t(t) = k_cal * [signal(t) - baseline]",
        "limitations": ["Calibration must be valid over the signal range and bandwidth.",
        "No coil geometry, mutual inductance, plasma-current contribution, ferromagnetic response, or absolute calibration is inferred.",
        "Pickup-coil voltage generally requires a validated transfer function or Faraday-law integration; this function does not integrate voltage."]}
    if signal_uncertainty is not None:
        sx = np.asarray(signal_uncertainty, dtype=float)
        if sx.shape != x.shape or not np.all(np.isfinite(sx)) or np.any(sx < 0):
            raise ValueError("signal_uncertainty must match signal and be finite/non-negative")
        out["B_toroidal_uncertainty_T"] = np.sqrt((k*sx)**2 + (corrected*sk)**2)
    elif sk > 0:
        out["B_toroidal_uncertainty_T"] = np.abs(corrected)*sk
    if pickup_voltage_V is not None:
        if pickup_sensitivity_V_per_T_per_s is None or initial_field_T is None:
            raise ValueError(
                "pickup_sensitivity_V_per_T_per_s and initial_field_T are required "
                "when pickup_voltage_V is supplied"
            )
        integrated = integrate_pickup_coil_voltage(
            t, pickup_voltage_V,
            sensitivity_V_per_T_per_s=pickup_sensitivity_V_per_T_per_s,
            initial_field_T=initial_field_T,
            voltage_baseline_V=pickup_voltage_baseline_V,
            voltage_uncertainty_V=pickup_voltage_uncertainty_V,
            sensitivity_uncertainty_V_per_T_per_s=pickup_sensitivity_uncertainty_V_per_T_per_s,
            initial_field_uncertainty_T=initial_field_uncertainty_T,
        )
        out["pickup_voltage_integration"] = integrated
        out["B_toroidal_integrated_T"] = integrated["B_toroidal_T"]
        out["delta_B_T"] = integrated["delta_B_T"]
        out["dB_dt_T_s"] = integrated["dB_dt_T_s"]
        out["B_toroidal_integrated_uncertainty_T"] = integrated.get("B_toroidal_uncertainty_T")
    return out


def integrate_pickup_coil_voltage(
    time_s, voltage_V, *, sensitivity_V_per_T_per_s: float,
    initial_field_T: float, voltage_baseline_V: float = 0.0,
    voltage_uncertainty_V=None, sensitivity_uncertainty_V_per_T_per_s: float = 0.0,
    initial_field_uncertainty_T: float = 0.0,
):
    """Recover field change from a calibrated pickup coil using Faraday-law integration.

    sensitivity_V_per_T_per_s is the validated system transfer coefficient for
    V = sensitivity * dB/dt over the sampled bandwidth. The initial field is
    required because integration determines field change, not its absolute offset.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(voltage_V, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or t.size != v.size or t.size < 2:
        raise ValueError("time_s and voltage_V must be 1-D arrays of equal length >= 2")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        raise ValueError("time_s and voltage_V must contain only finite values")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")
    s = float(sensitivity_V_per_T_per_s)
    b0 = float(initial_field_T)
    vb = float(voltage_baseline_V)
    ss = float(sensitivity_uncertainty_V_per_T_per_s)
    sb0 = float(initial_field_uncertainty_T)
    if not np.isfinite(s) or s == 0:
        raise ValueError("sensitivity_V_per_T_per_s must be finite and non-zero")
    if not all(np.isfinite(z) for z in (b0, vb, ss, sb0)) or ss < 0 or sb0 < 0:
        raise ValueError("Initial field/baseline must be finite and uncertainties non-negative")
    corrected = v - vb
    derivative = corrected / s
    increments = 0.5 * (derivative[:-1] + derivative[1:]) * dt
    delta_b = np.concatenate(([0.0], np.cumsum(increments)))
    field = b0 + delta_b
    out = {
        "time_s": t.copy(), "voltage_corrected_V": corrected,
        "dB_dt_T_s": derivative, "delta_B_T": delta_b, "B_toroidal_T": field,
        "initial_field_T": b0, "sensitivity_V_per_T_per_s": s,
        "voltage_baseline_V": vb,
        "integration_method": "cumulative trapezoidal integration",
        "physical_relation": "B(t)=B(t0)+integral[(V(t)-V_baseline)/sensitivity]dt",
        "sampling_rate_Hz": float(1.0 / np.median(dt)),
        "sampling_interval_jitter_fraction": float(np.std(dt) / np.mean(dt)),
        "limitations": [
            "Requires independently validated pickup-coil sensitivity/transfer function over the signal bandwidth.",
            "Absolute field depends on the supplied initial-field boundary condition.",
            "Baseline offset causes integration drift; no automatic detrending or frequency-response correction is applied.",
            "Uncertainty model assumes independent pointwise voltage noise and excludes correlated drift, baseline uncertainty, bandwidth error and model discrepancy."
        ]
    }
    if voltage_uncertainty_V is not None:
        sv = np.asarray(voltage_uncertainty_V, dtype=float)
        if sv.shape != v.shape or not np.all(np.isfinite(sv)) or np.any(sv < 0):
            raise ValueError("voltage_uncertainty_V must match voltage and be finite/non-negative")
        # Exact propagation for independent sample noise through cumulative
        # trapezoidal integration. Adjacent intervals share an interior voltage
        # sample, so their contributions are correlated even when the raw sample
        # errors are independent. Add the resulting cross terms at each interior
        # time point instead of treating interval integrals as independent.
        q = dt / (2.0 * s)
        interval_variance = q**2 * (sv[:-1]**2 + sv[1:]**2)
        cumulative_variance = np.concatenate(([0.0], np.cumsum(interval_variance)))
        if sv.size > 2:
            shared_sample_cross_terms = (
                2.0 * q[:-1] * q[1:] * sv[1:-1]**2
            )
            cumulative_variance[2:] += np.cumsum(shared_sample_cross_terms)
        # Sensitivity and initial-condition terms are propagated separately.
        sigma_voltage_integral = np.sqrt(cumulative_variance)
        sigma_sensitivity = np.abs(delta_b / s) * ss
        out["B_toroidal_uncertainty_T"] = np.sqrt(sb0**2 + sigma_voltage_integral**2 + sigma_sensitivity**2)
    elif ss > 0 or sb0 > 0:
        out["B_toroidal_uncertainty_T"] = np.sqrt(sb0**2 + (np.abs(delta_b / s) * ss)**2)
    return out
