from __future__ import annotations
import numpy as np


def analyze_runaway_electron_signatures(time_s, hard_xray_rate_s, *, threshold_sigma: float = 5.0, loop_voltage_V=None, plasma_current_A=None) -> dict:
    """Detect hard-X-ray bursts as candidate runaway-electron signatures."""
    t = np.asarray(time_s, dtype=float).ravel()
    hxr = np.asarray(hard_xray_rate_s, dtype=float).ravel()
    if t.size != hxr.size or t.size < 5 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(hxr)) or np.any(np.diff(t) <= 0):
        raise ValueError('Time and hard-X-ray rate must be finite, equal length and time strictly increasing.')
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError('threshold_sigma must be positive and finite.')
    baseline = float(np.median(hxr))
    noise = float(1.4826 * np.median(np.abs(hxr - baseline)))
    if noise <= np.finfo(float).eps:
        noise = float(np.std(hxr, ddof=0))
    threshold = baseline + threshold_sigma * noise
    active = hxr > threshold
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    events = []
    for start, stop in zip(starts, stops):
        peak = start + int(np.argmax(hxr[start:stop]))
        events.append({'start_time_s': float(t[start]), 'end_time_s': float(t[stop-1]),
            'duration_s': float(t[stop-1]-t[start]), 'peak_time_s': float(t[peak]),
            'peak_rate_s-1': float(hxr[peak]), 'peak_excess_over_baseline_s-1': float(hxr[peak]-baseline)})
    result = {'diagnostic': 'Runaway-electron diagnostics', 'baseline_hard_xray_rate_s-1': baseline,
        'robust_noise_sigma_s-1': noise, 'detection_threshold_s-1': threshold,
        'candidate_event_mask': active, 'candidate_events': events,
        'assumptions_and_limits': ['Hard-X-ray bursts are candidate signatures, not proof of runaway-electron generation.',
            'Detector background, dead time, bremsstrahlung sources and threshold choice affect detections.',
            'Runaway energy/distribution inference requires detector response and transport/radiation modeling.']}
    for name, values in [('loop_voltage_V', loop_voltage_V), ('plasma_current_A', plasma_current_A)]:
        if values is not None:
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size != t.size or not np.all(np.isfinite(arr)):
                raise ValueError(f'{name} must be finite and match time_s.')
            result[name] = arr
            result[name.replace('_', '_') + '_during_events'] = arr[active]
    return result
