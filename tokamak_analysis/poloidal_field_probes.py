"""Calibrated poloidal-field magnetic-probe signal analysis.

This module processes already digitized probe voltages/fields. It does not infer
coil currents, magnetic flux, or an equilibrium from local measurements.
"""
from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi


def analyze_poloidal_field_probes(
    time_s,
    signal,
    *,
    calibration_gain_T_per_unit=1.0,
    calibration_offset_T=0.0,
    signal_uncertainty=None,
    baseline_interval_s=None,
    probe_names=None,
):
    """Calibrate and quality-check one or more poloidal-field probe channels.

    Parameters
    ----------
    time_s : 1-D array
        Strictly increasing sample times in seconds.
    signal : 1-D or 2-D array
        Digitized signals, shape (samples,) or (samples, probes). Each channel
        must use a documented gain/offset; scalar values broadcast to channels.
    calibration_gain_T_per_unit, calibration_offset_T : scalar or 1-D array
        Calibration mapping Bp = gain * signal + offset, in tesla.
    signal_uncertainty : optional scalar or array
        Independent 1-sigma uncertainty in signal units; propagated through gain.
    baseline_interval_s : optional (start, stop)
        Time interval used to estimate and subtract each channel's mean baseline.
        Baseline correction is opt-in and is not a substitute for calibration.
    probe_names : optional sequence[str]
        Channel names.

    Returns a dictionary of calibrated signals, derivatives, basic statistics,
    optional uncertainty, and sampling diagnostics. The derivative is dBp/dt,
    not a spatial gradient or an inferred plasma current.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.ndim != 1 or t.size < 3 or not np.all(np.isfinite(t)):
        raise ValueError("time_s must be a finite 1-D array with at least 3 samples.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing.")
    if y.ndim == 1:
        y = y[:, None]
        squeeze = True
    elif y.ndim == 2:
        squeeze = False
    else:
        raise ValueError("signal must be a 1-D or 2-D array.")
    if y.shape[0] != t.size or y.shape[1] < 1 or not np.all(np.isfinite(y)):
        raise ValueError("signal must be finite and its first dimension must match time_s.")

    n_channels = y.shape[1]
    def channel_parameter(value, name):
        a = np.asarray(value, dtype=float)
        if a.ndim == 0:
            a = np.full(n_channels, float(a))
        if a.shape != (n_channels,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite scalar or one value per channel.")
        return a

    gain = channel_parameter(calibration_gain_T_per_unit, "calibration_gain_T_per_unit")
    offset = channel_parameter(calibration_offset_T, "calibration_offset_T")
    if np.any(gain == 0):
        raise ValueError("Calibration gains must be non-zero.")
    bp = y * gain[None, :] + offset[None, :]

    baseline_mask = None
    baseline_values = np.zeros(n_channels)
    if baseline_interval_s is not None:
        if len(baseline_interval_s) != 2:
            raise ValueError("baseline_interval_s must be a (start, stop) pair.")
        start, stop = map(float, baseline_interval_s)
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("Baseline interval must have finite start < stop.")
        baseline_mask = (t >= start) & (t <= stop)
        if np.count_nonzero(baseline_mask) < 2:
            raise ValueError("Baseline interval must include at least two samples.")
        baseline_values = np.mean(bp[baseline_mask], axis=0)
        bp = bp - baseline_values[None, :]

    dbp_dt = np.gradient(bp, t, axis=0, edge_order=2)
    dt = np.diff(t)
    dt_median = float(np.median(dt))
    jitter = float(np.max(np.abs(dt - dt_median)) / dt_median)
    out = {
        "time_s": t.copy(),
        "B_poloidal_T": bp[:, 0] if squeeze else bp,
        "dB_poloidal_dt_T_s": dbp_dt[:, 0] if squeeze else dbp_dt,
        "baseline_removed_T": baseline_values[0].item() if squeeze else baseline_values.copy(),
        "sample_interval_median_s": dt_median,
        "sampling_rate_median_Hz": 1.0 / dt_median,
        "sampling_interval_relative_max_deviation": jitter,
        "channel_mean_T": np.mean(bp, axis=0)[0].item() if squeeze else np.mean(bp, axis=0),
        "channel_rms_T": np.sqrt(np.mean(bp**2, axis=0))[0].item() if squeeze else np.sqrt(np.mean(bp**2, axis=0)),
        "channel_min_T": np.min(bp, axis=0)[0].item() if squeeze else np.min(bp, axis=0),
        "channel_max_T": np.max(bp, axis=0)[0].item() if squeeze else np.max(bp, axis=0),
        "physical_relations": {
            "calibration": "Bp(t) = gain * signal(t) + offset; optional baseline mean is then subtracted",
            "time_derivative": "dBp/dt via nonuniform-grid finite differences; not a spatial derivative",
            "faraday_law_context": "induced loop voltage is related to flux change by V_loop = -dPhi/dt; converting a probe derivative to flux/voltage requires the actual probe geometry and transfer calibration",
        },
        "limitations": [
            "Requires calibrated gain and offset appropriate to the probe, electronics, frequency response, and acquisition chain.",
            "Local Bp measurements do not uniquely determine poloidal flux, PF-coil currents, plasma boundary, or equilibrium.",
            "Derivative noise can be amplified; filtering must be selected and documented for the signal bandwidth.",
            "Sampling diagnostics do not prove the signal is alias-free; compare sample rate with analog bandwidth and the highest frequency of interest.",
        ],
    }
    if probe_names is not None:
        names = list(probe_names)
        if len(names) != n_channels or any(not isinstance(x, str) or not x.strip() for x in names):
            raise ValueError("probe_names must contain one non-empty name per channel.")
        out["probe_names"] = names

    if signal_uncertainty is not None:
        u = np.asarray(signal_uncertainty, dtype=float)
        if u.ndim == 0:
            u = np.full_like(y, float(u))
        elif u.ndim == 1 and squeeze and u.shape == (t.size,):
            u = u[:, None]
        elif u.ndim == 1 and not squeeze and u.shape == (n_channels,):
            u = np.broadcast_to(u[None, :], y.shape)
        if u.shape != y.shape or not np.all(np.isfinite(u)) or np.any(u < 0):
            raise ValueError("signal_uncertainty must be non-negative, finite, and broadcastable to signal.")
        ub = u * np.abs(gain)[None, :]
        out["B_poloidal_uncertainty_T"] = ub[:, 0] if squeeze else ub
        # Independent-sample uncertainty for a three-point central difference.
        # Edge points use NumPy's one-sided second-order gradient coefficients.
        coeff = np.gradient(np.eye(t.size), t, axis=0, edge_order=2)
        ud = np.sqrt((coeff**2) @ (ub**2))
        out["dB_poloidal_dt_uncertainty_T_s"] = ud[:, 0] if squeeze else ud

    if jitter > 0.01:
        out["sampling_warning"] = "Sampling intervals vary by more than 1%; frequency-domain analyses should account for nonuniform sampling."
    out["caveat"] = "This workflow starts from digitized probe signals and calibrated coefficients; it is not an equilibrium solver."
    return out
