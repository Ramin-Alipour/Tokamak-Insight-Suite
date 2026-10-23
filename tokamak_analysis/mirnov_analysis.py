"""Mirnov / magnetic pickup-coil analysis with explicit calibration limits.

The built-in scope is signal-quality/statistical analysis, Welch PSD and peak
candidates, ideal fixed-coil Faraday conversion, voltage integration to relative
magnetic perturbation, and coherence-weighted toroidal mode-number candidates.
It is not an equilibrium solver or a substitute for coil transfer-function and
absolute calibration.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import coherence, csd, detrend as scipy_detrend, find_peaks, welch


def _signal(time_s: Any, values: Any, minimum: int = 2):
    """Validate one time series without silently sorting or repairing samples."""
    t, y = np.asarray(time_s, dtype=float), np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError("time and signal must be matching one-dimensional arrays")
    if len(t) < minimum or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError(f"At least {minimum} finite matching samples are required")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing")
    return t, y


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(x) or x < 1 or not x.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(x)


def _dt(t: np.ndarray) -> float:
    d = np.diff(t)
    dt = float(np.median(d))
    if not np.isfinite(dt) or dt <= 0 or not np.allclose(
        d, dt, rtol=1e-3, atol=max(1e-15, dt * 1e-12)
    ):
        raise ValueError("Uniform sampling is required for spectral analysis")
    return dt


def _frequency_band(frequency_range_Hz, nyquist: float):
    if frequency_range_Hz is None:
        return 0.0, nyquist
    try:
        low, high = map(float, frequency_range_Hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("frequency_range_Hz must be a (low, high) pair") from exc
    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or low < 0
        or high <= low
        or high > nyquist * (1 + 1e-12)
    ):
        raise ValueError("frequency range must satisfy 0 <= low < high <= Nyquist")
    return low, min(high, nyquist)


def analyze_mirnov_signal(
    time_s,
    voltage_V,
    *,
    coil_area_m2=None,
    turns=None,
    detrend="constant",
    nperseg=None,
    peak_prominence_fraction=0.01,
    frequency_range_Hz=None,
):
    """Analyze one calibrated-time Mirnov voltage trace.

    PSD uses Welch density scaling and therefore has units V²/Hz. The ideal
    fixed-coil Faraday law is V=-N*A*dB/dt; geometry alone does not correct the
    frequency response, orientation, integrator, or digitizer calibration.
    Returned spectral peaks are candidates, not MHD-mode identifications.
    """
    t, v = _signal(time_s, voltage_V, minimum=8)
    dt = _dt(t)
    if detrend not in {"constant", "linear", "none"}:
        raise ValueError("detrend must be 'constant', 'linear', or 'none'")
    try:
        prominence = float(peak_prominence_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("peak_prominence_fraction must be finite and in [0, 1]") from exc
    if not np.isfinite(prominence) or not 0 <= prominence <= 1:
        raise ValueError("peak_prominence_fraction must be finite and in [0, 1]")
    n = min(1024, len(v)) if nperseg is None else _integer(nperseg, "nperseg")
    if n > len(v):
        raise ValueError("nperseg cannot exceed sample count")
    fs = 1.0 / dt
    nyquist = 0.5 * fs
    low, high = _frequency_band(frequency_range_Hz, nyquist)
    y = (
        v - np.mean(v)
        if detrend == "constant"
        else scipy_detrend(v, type="linear")
        if detrend == "linear"
        else v.copy()
    )
    f, p = welch(y, fs=fs, nperseg=n, detrend=False, scaling="density")
    band = (f >= low) & (f <= high)
    if not np.any(band):
        raise ValueError("frequency range contains no Welch frequency bins")
    peaks, _ = find_peaks(p)
    peaks = peaks[band[peaks]]
    max_p = float(np.max(p[band]))
    if max_p > 0:
        peaks = peaks[p[peaks] >= prominence * max_p]
    else:
        peaks = np.array([], dtype=int)
    peaks = peaks[np.argsort(p[peaks])[::-1]][:8]
    out = {
        "time_s": t,
        "voltage_V": v,
        "sampling_interval_s": dt,
        "sampling_frequency_Hz": fs,
        "nyquist_frequency_Hz": nyquist,
        "frequency_Hz": f,
        "power_spectral_density": p,
        "power_spectral_density_V2_per_Hz": p,
        "dominant_peaks_Hz": f[peaks],
        "dominant_peak_psd_V2_per_Hz": p[peaks],
        "mean_voltage_V": float(np.mean(v)),
        "rms_voltage_V": float(np.sqrt(np.mean(v**2))),
        "ac_rms_voltage_V": float(np.std(v)),
        "peak_to_peak_voltage_V": float(np.ptp(v)),
        "detrend_method": detrend,
        "welch_nperseg": n,
        "frequency_range_for_peaks_Hz": (low, high),
        "equation": "Welch PSD; ideal fixed-coil relation V=-N*A*dB/dt",
        "caveat": (
            "No transfer-function, orientation, pickup, absolute-field or "
            "calibration correction is automatic; PSD peaks alone do not "
            "identify MHD modes."
        ),
    }
    if coil_area_m2 is not None or turns is not None:
        if coil_area_m2 is None or turns is None:
            raise ValueError("coil_area_m2 and turns must be supplied together")
        if isinstance(coil_area_m2, (bool, np.bool_)):
            raise ValueError("coil area must be finite and positive")
        try:
            area = float(coil_area_m2)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coil area must be finite and positive") from exc
        count = _integer(turns, "turns")
        if not np.isfinite(area) or area <= 0:
            raise ValueError("coil area must be finite and positive")
        out.update(
            {
                "approx_Bdot_T_s": -v / (count * area),
                "approx_Bdot_detrended_T_s": -y / (count * area),
                "coil_area_m2": area,
                "turns": count,
            }
        )
    return out


def magnetic_perturbation_from_mirnov_voltage(
    time_s,
    coil_voltage_V,
    turns,
    coil_area_m2,
    *,
    initial_delta_B_T=0.0,
    baseline_voltage_V=0.0,
    polarity_sign=-1,
):
    """Integrate coil voltage to relative delta-B using trapezoidal integration.

    A constant baseline can be removed explicitly. The caller must choose the
    initial perturbation and polarity and assess drift/low-frequency response.
    """
    t, v = _signal(time_s, coil_voltage_V)
    count = _integer(turns, "turns")
    try:
        area = float(coil_area_m2)
        b0 = float(initial_delta_B_T)
        baseline = float(baseline_voltage_V)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("area, initial field and baseline must be finite numbers") from exc
    if not np.isfinite(area) or area <= 0:
        raise ValueError("coil_area_m2 must be finite and positive")
    if not np.isfinite(b0) or not np.isfinite(baseline):
        raise ValueError("initial field and baseline must be finite")
    if isinstance(polarity_sign, (bool, np.bool_)) or polarity_sign not in (-1, 1):
        raise ValueError("polarity_sign must be -1 or +1")
    dbdt = int(polarity_sign) * (v - baseline) / (count * area)
    db = np.r_[b0, b0 + np.cumsum(0.5 * (dbdt[1:] + dbdt[:-1]) * np.diff(t))]
    return {
        "time_s": t,
        "delta_B_T": db,
        "dB_dt_T_s": dbdt,
        "turns": count,
        "coil_area_m2": area,
        "baseline_voltage_V": baseline,
        "polarity_sign": int(polarity_sign),
        "equation": "dB/dt=polarity_sign*(V-baseline)/(N*A)",
        "caveat": (
            "Returns perturbation relative to supplied initial value, not absolute "
            "field; integration drift, offsets, transfer response and orientation "
            "must be validated."
        ),
    }


def analyze_mirnov_array(
    time_s,
    voltage_channels_V,
    toroidal_angles_rad,
    *,
    reference_channel=0,
    nperseg=None,
):
    """Estimate cross-channel phase/coherence and alias-prone toroidal mode.

    Input voltages have shape (channels, samples); angles are toroidal sensor
    locations in radians. The estimate is a candidate at the strongest
    non-DC reference-channel frequency, not an automatic mode classification.
    """
    t = np.asarray(time_s, dtype=float)
    signals = np.asarray(voltage_channels_V, dtype=float)
    angles = np.asarray(toroidal_angles_rad, dtype=float)
    if t.ndim != 1 or signals.ndim != 2 or signals.shape[1] != len(t):
        raise ValueError("voltage_channels_V must have shape (n_channels, len(time_s))")
    channels = signals.shape[0]
    if channels < 3:
        raise ValueError("At least three Mirnov channels are required")
    if angles.shape != (channels,) or not np.all(np.isfinite(angles)):
        raise ValueError("One finite toroidal angle is required per channel")
    t, _ = _signal(t, signals[0], minimum=8)
    dt = _dt(t)
    if not np.all(np.isfinite(signals)):
        raise ValueError("channel signals must be finite")
    if isinstance(reference_channel, (bool, np.bool_)) or not isinstance(
        reference_channel, (int, np.integer)
    ) or not 0 <= reference_channel < channels:
        raise ValueError("reference_channel must be a valid channel index")
    # Reject duplicate locations modulo 2*pi (including 0 and 2*pi).
    wrapped_angles = np.mod(angles, 2 * np.pi)
    sorted_angles = np.sort(wrapped_angles)
    circular_gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2 * np.pi])
    if np.any(circular_gaps <= 1e-10):
        raise ValueError("toroidal angles must be unique modulo 2*pi")
    n = min(1024, len(t)) if nperseg is None else _integer(nperseg, "nperseg")
    if n > len(t):
        raise ValueError("nperseg cannot exceed sample count")
    centered = signals - np.mean(signals, axis=1, keepdims=True)
    fs = 1.0 / dt
    f, psd = welch(centered[reference_channel], fs=fs, nperseg=n)
    if len(f) < 2 or not np.any(psd[1:] > 0):
        raise ValueError("Reference channel has no non-DC spectral power")
    k = int(np.argmax(psd[1:]) + 1)
    phase, coh = [], []
    for ch in range(channels):
        _, cross = csd(centered[reference_channel], centered[ch], fs=fs, nperseg=n)
        _, c = coherence(centered[reference_channel], centered[ch], fs=fs, nperseg=n)
        phase.append(float(np.angle(cross[k])))
        coh.append(float(np.clip(c[k], 0.0, 1.0)))
    phase, coh = np.asarray(phase), np.asarray(coh)
    relative_angles = np.angle(np.exp(1j * (angles - angles[reference_channel])))
    valid = (np.arange(channels) != reference_channel) & (coh > 1e-8)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Insufficient non-reference channels with usable coherence")
    candidates = np.arange(-channels, channels + 1, dtype=int)
    scores = []
    for mode in candidates:
        residual = np.angle(np.exp(1j * (phase[valid] - mode * relative_angles[valid])))
        scores.append(float(np.sqrt(np.average(residual**2, weights=coh[valid]))))
    scores = np.asarray(scores)
    best = int(np.argmin(scores))
    # A coarse diagnostic for the finite sensor-array sampling; this is not a
    # guarantee of uniqueness for irregular geometry or mixed modes.
    min_spacing = float(np.min(circular_gaps))
    alias_limit = int(np.floor(np.pi / min_spacing)) if min_spacing > 0 else 0
    return {
        "frequency_Hz": f,
        "reference_psd_V2_per_Hz": psd,
        "dominant_frequency_Hz": float(f[k]),
        "relative_phase_rad": phase,
        "coherence": coh,
        "toroidal_angles_rad": angles,
        "candidate_mode_numbers": candidates,
        "candidate_phase_rms_rad": scores,
        "estimated_toroidal_mode_number": int(candidates[best]),
        "phase_fit_rms_rad": float(scores[best]),
        "minimum_toroidal_sensor_spacing_rad": min_spacing,
        "approximate_mode_alias_limit": alias_limit,
        "reference_channel": int(reference_channel),
        "sampling_frequency_Hz": fs,
        "nyquist_frequency_Hz": fs / 2,
        "welch_nperseg": n,
        "caveat": (
            "Mode number is an alias-prone candidate at one frequency. The spacing "
            "limit is only a coarse sampling diagnostic; verify polarity, coil "
            "geometry, phase calibration, coherence, bandwidth, mode structure "
            "and competing modes independently."
        ),
    }
