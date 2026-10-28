"""Electron-cyclotron-emission (ECE) channel and profile analysis.

This module deliberately separates resonance mapping from temperature inference:
frequency-to-radius mapping requires an equilibrium B(R) model, while brightness
temperature is only a plasma-temperature estimate in the optically thick,
thermalized regime.
"""
from __future__ import annotations

import numpy as np

from scipy.constants import elementary_charge as E_CHARGE, electron_mass as M_E, speed_of_light as C_LIGHT, Boltzmann as K_B
from scipy.signal import welch as welch_psd, get_window, csd as cross_spectral_density, coherence as signal_coherence, find_peaks, fftconvolve
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid
from scipy.stats import chi2


def resonance_field_T(frequency_Hz, harmonic: int = 1):
    """Magnetic field [T] for non-relativistic ECE resonance at frequency [Hz]."""
    f = np.asarray(frequency_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must contain finite positive values.")
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)) or harmonic < 1:
        raise ValueError("harmonic must be a positive integer.")
    result = 2.0 * np.pi * M_E * f / (harmonic * E_CHARGE)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Resonance-field calculation overflowed or left the representable physical range.")
    return float(result) if result.ndim == 0 else result



def resonance_field_relativistic_T(frequency_Hz, harmonic: int = 1, *,
                                   electron_energy_eV=0.0, parallel_beta=0.0):
    """Monoenergetic relativistic/Doppler resonance estimate, not a full ECE solver.

    Uses f = n*f_ce/[gamma*(1-beta_parallel)]; beta_parallel is signed under
    this convention. Distribution-averaged emissivity and ray tracing are not included.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    energy = np.asarray(electron_energy_eV, dtype=float)
    beta = np.asarray(parallel_beta, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must contain finite positive values.")
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)) or harmonic < 1:
        raise ValueError("harmonic must be a positive integer.")
    if np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("electron_energy_eV must be finite and non-negative.")
    if np.any(~np.isfinite(beta)) or np.any(np.abs(beta) >= 1):
        raise ValueError("parallel_beta must be finite and strictly between -1 and 1.")
    with np.errstate(over="ignore", invalid="ignore"):
        gamma = 1.0 + energy * E_CHARGE / (M_E * C_LIGHT**2)
        result = resonance_field_T(f, harmonic) * gamma * (1.0 - beta)
    if np.any(~np.isfinite(gamma)) or np.any(gamma < 1.0) or np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Relativistic resonance calculation overflowed or left the representable physical range.")
    return float(result) if np.ndim(result) == 0 else result


def resonance_radius_m(frequency_Hz, major_radius_m, B_axis_T, *,
                       harmonic: int = 1, sign: int = 1):
    """Invert a circular toroidal-field model B(R)=B_axis*R_axis/R.

    Returns the resonance major radius for B(R)=B_axis*R_axis/R.

    A frequency and a one-dimensional toroidal-field model determine the
    major-radius coordinate; they do not determine which side of the plasma a
    line of sight intersects. The legacy ``sign`` argument is therefore
    accepted only at its default (+1); a mirrored high-field-side radius would
    be physically unjustified without viewing geometry.
    This simplified model is not a substitute for EFIT mapping in shaped plasmas.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    r0 = np.asarray(major_radius_m, dtype=float)
    b0 = np.asarray(B_axis_T, dtype=float)
    if np.any(~np.isfinite(r0)) or np.any(r0 <= 0):
        raise ValueError("major_radius_m must be finite and positive.")
    if np.any(~np.isfinite(b0)) or np.any(b0 <= 0):
        raise ValueError("B_axis_T must be finite and positive.")
    if isinstance(sign, (bool, np.bool_)) or sign != 1:
        raise ValueError(
            "sign=-1 cannot be represented by a 1-D major-radius mapping; "
            "supply viewing geometry and equilibrium for high/low-field-side localization."
        )
    b_res = resonance_field_T(f, harmonic)
    radius = r0 * b0 / b_res
    return float(radius) if np.ndim(radius) == 0 else radius


def brightness_temperature_K(spectral_radiance_W_m2_sr_Hz, frequency_Hz):
    """Rayleigh-Jeans brightness temperature [K] from calibrated spectral radiance."""
    radiance = np.asarray(spectral_radiance_W_m2_sr_Hz, dtype=float)
    freq = np.asarray(frequency_Hz, dtype=float)
    if np.any(~np.isfinite(radiance)) or np.any(radiance < 0):
        raise ValueError("spectral radiance must be finite and non-negative.")
    if np.any(~np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("frequency_Hz must be finite and positive.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tb = C_LIGHT**2 * radiance / (2.0 * K_B * freq**2)
    if np.any(~np.isfinite(tb)) or np.any(tb < 0):
        raise ValueError("Brightness-temperature calculation overflowed; check physical units and numeric range.")
    return float(tb) if tb.ndim == 0 else tb



def convolve_ece_receiver_response(frequency_Hz, spectral_radiance_W_m2_sr_Hz,
                                   response_frequency_Hz, response_weight):
    """Apply a user-supplied receiver spectral response to calibrated radiance.

    response_weight has shape (channel, response_frequency) and is integrated
    against the supplied spectrum by linear interpolation. Each response row
    is normalized by its trapezoidal integral; no detector calibration or
    response-function inference is performed.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    radiance = np.asarray(spectral_radiance_W_m2_sr_Hz, dtype=float)
    rf = np.asarray(response_frequency_Hz, dtype=float)
    weights = np.asarray(response_weight, dtype=float)
    if f.ndim != 1 or f.size < 2 or np.any(~np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("frequency_Hz must be finite, positive, strictly increasing, and contain at least two points.")
    if (radiance.ndim < 1 or any(size == 0 for size in radiance.shape)
            or radiance.shape[-1:] != f.shape or np.any(~np.isfinite(radiance)) or np.any(radiance < 0)):
        raise ValueError("spectral_radiance_W_m2_sr_Hz must be a non-empty finite, non-negative array whose final axis matches frequency_Hz.")
    if rf.ndim != 1 or rf.size < 2 or np.any(~np.isfinite(rf)) or np.any(rf <= 0) or np.any(np.diff(rf) <= 0):
        raise ValueError("response_frequency_Hz must be finite, positive, strictly increasing, and contain at least two points.")
    if weights.ndim != 2 or weights.shape[1] != rf.size or weights.shape[0] < 1 or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("response_weight must be a finite non-negative (channel, response-frequency) matrix.")
    if rf[0] < f[0] or rf[-1] > f[-1]:
        raise ValueError("Receiver response frequencies must lie within the measured radiance spectrum; extrapolation is forbidden.")
    integrals = trapezoid(weights, rf, axis=1)
    if np.any(~np.isfinite(integrals)) or np.any(integrals <= 0):
        raise ValueError("Every receiver-response channel must have positive integrated weight (finite).")
    normalized = weights / integrals[:, None]
    effective_center_frequency = trapezoid(weights * rf[None, :], rf, axis=1) / integrals
    sampled = np.asarray([np.interp(rf, f, row) for row in radiance.reshape(-1, f.size)])
    sampled = sampled.reshape(*radiance.shape[:-1], rf.size)
    # Use trapezoidal quadrature, including for irregular response grids.
    effective = np.stack([
        trapezoid(sampled * normalized[channel], rf, axis=-1)
        for channel in range(weights.shape[0])
    ], axis=-1)
    if np.any(~np.isfinite(effective)) or np.any(effective < 0):
        raise ValueError("Receiver-response convolution produced non-finite or negative radiance.")
    return {
        "channel_spectral_radiance_W_m2_sr_Hz": effective,
        "response_frequency_Hz": rf,
        "normalized_response_weight": normalized,
        "effective_center_frequency_Hz": effective_center_frequency,
        "response_channel_count": int(weights.shape[0]),
        "response_convolution_performed": True,
        "calibration_performed": False,
        "limitations": ["Uses the user-supplied response kernel and linear interpolation of the supplied calibrated spectrum.", "Does not perform radiative-transfer inversion or infer missing spectral coverage."],
    }


def brightness_temperature_eV(spectral_radiance_W_m2_sr_Hz, frequency_Hz):
    """Brightness temperature expressed as an energy [eV], not a guaranteed Te."""
    result = K_B / E_CHARGE * brightness_temperature_K(
        spectral_radiance_W_m2_sr_Hz, frequency_Hz
    )
    return float(result) if np.ndim(result) == 0 else result


def analyze_ece(frequency_Hz, *, signal=None, time_s=None,
                spectral_radiance_W_m2_sr_Hz=None, major_radius_m=None,
                B_axis_T=None, harmonic=2, channel_bandwidth_Hz=None,
                noise_std=None, optical_depth=None, psd_confidence_level=0.95,
                mode="brightness_temperature", welch_nperseg=None,
                compute_cross_channel_spectra=True, stationarity_segments=4,
                peak_prominence_fraction=0.1, electron_energy_eV=None, parallel_beta=0.0):
    """Analyze calibrated ECE channels and return validated derived quantities.

    signal may be a (time, channel) fluctuation matrix. Temperature inference
    requires calibrated spectral radiance and is reported as brightness
    temperature; interpreting it as Te requires optical thickness/thermalization.
    Fluctuation mode returns a full-record periodogram and Welch PSD with
    approximate confidence bounds; a supplied noise RMS yields a white-noise
    PSD reference only, never automatic noise subtraction.
    """

    f = np.asarray(frequency_Hz, dtype=float)
    if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be a non-empty finite positive 1-D array.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequency_Hz must be strictly increasing.")
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)) or harmonic < 1:
        raise ValueError("harmonic must be a positive integer.")
    if not isinstance(mode, str) or mode not in {"brightness_temperature", "channel_fluctuations", "resonance_mapping"}:
        raise ValueError("Unsupported mode.")
    if mode == "resonance_mapping" and (spectral_radiance_W_m2_sr_Hz is not None or signal is not None or time_s is not None):
        raise ValueError("resonance_mapping accepts resonance inputs only; omit radiance, signal, and time_s.")
    if mode == "brightness_temperature" and (signal is not None or time_s is not None):
        raise ValueError("brightness_temperature does not consume signal or time_s; use channel_fluctuations for time series.")
    if mode == "channel_fluctuations" and spectral_radiance_W_m2_sr_Hz is not None:
        raise ValueError("channel_fluctuations does not consume spectral radiance; use brightness_temperature mode.")
    if mode != "brightness_temperature" and optical_depth is not None:
        raise ValueError("optical_depth is only consumed in brightness_temperature mode.")
    if mode != "channel_fluctuations" and noise_std is not None:
        raise ValueError("noise_std is only consumed in channel_fluctuations mode.")
    if mode == "channel_fluctuations":
        try:
            psd_confidence_level = float(psd_confidence_level)
        except (TypeError, ValueError) as exc:
            raise ValueError("psd_confidence_level must be finite and strictly between 0 and 1.") from exc
        if not np.isfinite(psd_confidence_level) or not 0.0 < psd_confidence_level < 1.0:
            raise ValueError("psd_confidence_level must be finite and strictly between 0 and 1.")
        if welch_nperseg is not None and (
            isinstance(welch_nperseg, (bool, np.bool_)) or
            not isinstance(welch_nperseg, (int, np.integer)) or welch_nperseg < 8
        ):
            raise ValueError("welch_nperseg must be an integer >= 8.")
        if isinstance(stationarity_segments, (bool, np.bool_)) or not isinstance(stationarity_segments, (int, np.integer)) or stationarity_segments < 2:
            raise ValueError("stationarity_segments must be an integer >= 2.")
        if not isinstance(compute_cross_channel_spectra, (bool, np.bool_)):
            raise ValueError("compute_cross_channel_spectra must be boolean.")
        try:
            peak_prominence_fraction = float(peak_prominence_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("peak_prominence_fraction must be finite and between 0 and 1.") from exc
        if not np.isfinite(peak_prominence_fraction) or not 0 <= peak_prominence_fraction < 1:
            raise ValueError("peak_prominence_fraction must be finite and between 0 and 1.")
    if electron_energy_eV is None:
        resonance_fields = resonance_field_T(f, harmonic)
        resonance_model = "cold non-relativistic resonance"
    else:
        resonance_fields = resonance_field_relativistic_T(
            f, harmonic, electron_energy_eV=electron_energy_eV,
            parallel_beta=parallel_beta,
        )
        resonance_model = "monoenergetic relativistic/Doppler correction; not distribution averaged"
    out = {"frequency_Hz": f.copy(), "harmonic": int(harmonic),
           "resonance_field_T": resonance_fields,
           "resonance_model": resonance_model,
           "assumptions": [resonance_model, "frequency values must be in Hz; calibrated input units remain the user's responsibility"]}
    if channel_bandwidth_Hz is not None:
        bw = np.asarray(channel_bandwidth_Hz, dtype=float)
        if bw.ndim == 0 or bw.shape == (1,):
            bw = np.full(f.shape, float(bw.reshape(-1)[0]))
        if bw.shape != f.shape or np.any(~np.isfinite(bw)) or np.any(bw <= 0):
            raise ValueError("channel_bandwidth_Hz must be positive and scalar or match frequency_Hz.")
        if np.any(bw >= 2.0 * f):
            raise ValueError("Each channel bandwidth must be smaller than twice its center frequency.")
        out["channel_bandwidth_Hz"] = bw
        out["fractional_bandwidth"] = bw / f
    if major_radius_m is not None or B_axis_T is not None:
        if major_radius_m is None or B_axis_T is None:
            raise ValueError("major_radius_m and B_axis_T must be provided together.")
        r0 = np.asarray(major_radius_m, dtype=float)
        b0 = np.asarray(B_axis_T, dtype=float)
        if r0.ndim != 0 or not np.isfinite(r0) or r0 <= 0:
            raise ValueError("major_radius_m must be a finite positive scalar.")
        if b0.ndim != 0 or not np.isfinite(b0) or b0 <= 0:
            raise ValueError("B_axis_T must be a finite positive scalar.")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            radius = float(r0) * float(b0) / np.asarray(out["resonance_field_T"], dtype=float)
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("Resonance-radius mapping produced non-finite or non-positive radii.")
        out["resonance_radius_m"] = float(radius) if radius.ndim == 0 else radius
        out["equilibrium_model"] = "circular toroidal field B(R)=B_axis*R_axis/R"
        out["limitations"] = ["not EFIT mapping; ignores shaping and poloidal field"]
        if electron_energy_eV is None:
            out["limitations"].append("relativistic and Doppler shifts are not included")
        else:
            out["limitations"].append("relativistic/Doppler term is monoenergetic and approximate; distribution effects and line-of-sight geometry are not modeled")
    if mode == "brightness_temperature":
        if spectral_radiance_W_m2_sr_Hz is None:
            raise ValueError("Calibrated spectral_radiance_W_m2_sr_Hz is required for temperature mode.")
        rad = np.asarray(spectral_radiance_W_m2_sr_Hz, dtype=float)
        if rad.ndim == 0 or rad.shape[-1:] != f.shape:
            raise ValueError("spectral_radiance_W_m2_sr_Hz must have a final channel dimension matching frequency_Hz.")
        if np.any(~np.isfinite(rad)) or np.any(rad < 0):
            raise ValueError("spectral_radiance_W_m2_sr_Hz must contain finite non-negative values.")
        out["brightness_temperature_K"] = brightness_temperature_K(rad, f)
        out["brightness_temperature_eV"] = brightness_temperature_eV(rad, f)
        out["temperature_interpretation"] = (
            "May approximate electron temperature only where the emission is optically thick "
            "and the electron distribution is thermalized."
        )
        if optical_depth is not None:
            tau = np.asarray(optical_depth, dtype=float)
            if np.any(~np.isfinite(tau)) or np.any(tau < 0):
                raise ValueError("optical_depth must be finite and non-negative.")
            if tau.ndim > 1 or (tau.ndim == 1 and tau.shape not in {(1,), f.shape}):
                raise ValueError("optical_depth must be a scalar, one value, or one value per frequency channel.")
            if tau.shape == (1,):
                tau = np.full(f.shape, float(tau[0]))
            out["optical_depth"] = tau
            out["optical_depth_warning"] = "Tau alone does not establish thermalization or calibration validity."
            out["optical_depth_inversion_sensitivity_dTe_dTb"] = (
                None if np.any(tau <= 0) else 1.0 / (-np.expm1(-tau))
            )
            out["optical_depth_condition_warning"] = bool(np.any(tau < 1e-3))
            if out["optical_depth_condition_warning"]:
                out["optical_depth_condition_message"] = (
                    "At least one optical depth is below 1e-3; the single-slab Te inversion is strongly "
                    "ill-conditioned and highly sensitive to Tb, background and tau uncertainty."
                )
    if mode == "channel_fluctuations":
        if signal is None or time_s is None:
            raise ValueError("signal and time_s are required for channel_fluctuations mode.")
        y = np.asarray(signal, dtype=float)
        t = np.asarray(time_s, dtype=float)
        if y.ndim != 2 or y.shape != (t.size, f.size):
            raise ValueError("signal must have shape (len(time_s), len(frequency_Hz)).")
        if t.ndim != 1 or t.size < 3 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("time_s must be finite, strictly increasing, and contain at least 3 samples.")
        if np.any(~np.isfinite(y)):
            raise ValueError("signal must contain only finite samples.")
        if y.shape[0] < 3:
            raise ValueError("signal must contain at least 3 time samples.")
        dt = np.diff(t)
        if not np.allclose(dt, np.median(dt), rtol=1e-3, atol=0):
            raise ValueError("Sampling must be approximately uniform for FFT-based spectra.")
        y0 = y - np.mean(y, axis=0, keepdims=True)
        freq = np.fft.rfftfreq(t.size, d=float(np.median(dt)))
        fft = np.fft.rfft(y0, axis=0)
        psd = (np.abs(fft) ** 2) * (float(np.median(dt)) / t.size)
        # One-sided periodogram: double positive-frequency bins only, excluding
        # DC and (for even N) the Nyquist bin. This normalization satisfies
        # Parseval: integral(PSD df) equals the mean-square fluctuation.
        if t.size % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        fs = 1.0 / float(np.median(dt))
        # Welch averaging provides a lower-variance PSD estimate alongside the
        # full-record periodogram retained for backward compatibility.
        welch_nperseg = min(256 if welch_nperseg is None else int(welch_nperseg), t.size)
        welch_noverlap = welch_nperseg // 2
        welch_step = welch_nperseg - welch_noverlap
        welch_frequency, welch_density = welch_psd(
            y0, fs=fs, window="hann", nperseg=welch_nperseg,
            noverlap=welch_noverlap, detrend=False, scaling="density",
            axis=0, return_onesided=True,
        )
        # Approximate effective degrees of freedom account for correlated
        # overlapping Welch segments; confidence bounds are asymptotic.
        segment_count = 1 + (t.size - welch_nperseg) // welch_step
        window_values = get_window("hann", welch_nperseg, fftbins=True)
        window_energy = float(np.dot(window_values, window_values))
        overlap_terms = []
        for segment_lag in range(1, segment_count):
            shift = segment_lag * welch_step
            if shift >= welch_nperseg:
                break
            rho = float(np.dot(window_values[:-shift], window_values[shift:]) / window_energy)
            overlap_terms.append((1.0 - segment_lag / segment_count) * rho**2)
        welch_dof = float(2.0 * segment_count / (1.0 + 2.0 * sum(overlap_terms)))
        alpha = 1.0 - float(psd_confidence_level)
        chi2_lower = float(chi2.ppf(alpha / 2.0, welch_dof))
        chi2_upper = float(chi2.ppf(1.0 - alpha / 2.0, welch_dof))
        welch_ci_lower = welch_dof * welch_density / chi2_upper
        welch_ci_upper = welch_dof * welch_density / chi2_lower
        out.update({"time_s": t, "fluctuation_rms": np.sqrt(np.mean(y0**2, axis=0)),
                    "fluctuation_psd_frequency_Hz": freq, "fluctuation_psd": psd,
                    "welch_psd_frequency_Hz": welch_frequency,
                    "welch_psd": welch_density,
                    "welch_psd_confidence_level": float(psd_confidence_level),
                    "welch_psd_degrees_of_freedom_approx": welch_dof,
                    "welch_psd_confidence_lower": welch_ci_lower,
                    "welch_psd_confidence_upper": welch_ci_upper,
                    "welch_psd_segment_count": int(segment_count),
                    "welch_psd_nperseg": welch_nperseg,
                    "welch_psd_window": "hann",
                    "welch_psd_overlap_samples": welch_noverlap,
                    "sampling_frequency_Hz": fs,
                    "nyquist_frequency_Hz": 0.5 * fs,
                    "sampling_interval_median_s": float(np.median(dt)),
                    "sampling_interval_max_deviation_fraction": float(np.max(np.abs(dt - np.median(dt))) / np.median(dt)),
                    "sampling_uniformity_tolerance": 1e-3,
                    "aliasing_assessment": "Not determined from samples alone; verify analogue anti-alias filtering and receiver bandwidth externally.",
                    "stationarity_assessment": "Diagnostic only: split-record mean/RMS consistency; does not prove plasma stationarity."})
        chunks = np.array_split(y0, min(int(stationarity_segments), t.size))
        chunk_rms = np.asarray([np.sqrt(np.mean(chunk**2, axis=0)) for chunk in chunks])
        chunk_means = np.asarray([np.mean(chunk, axis=0) for chunk in chunks])
        rms_floor = np.maximum(np.sqrt(np.mean(y0**2, axis=0)), np.finfo(float).tiny)
        out["stationarity_chunk_rms_relative_range"] = np.ptp(chunk_rms, axis=0) / rms_floor
        out["stationarity_chunk_mean_relative_range"] = np.ptp(chunk_means, axis=0) / rms_floor
        out["stationarity_flag"] = bool(np.any(out["stationarity_chunk_rms_relative_range"] > 0.5) or np.any(out["stationarity_chunk_mean_relative_range"] > 0.5))
        peak_rows = []
        for channel in range(f.size):
            density = welch_density[:, channel]
            if density.size < 3 or not np.any(density[1:] > 0):
                continue
            peaks, properties = find_peaks(
                density, prominence=max(float(np.max(density)) * peak_prominence_fraction, np.finfo(float).tiny)
            )
            for peak, prominence in zip(peaks, properties.get("prominences", [])):
                if peak <= 0:
                    continue
                peak_rows.append({
                    "channel_index": int(channel),
                    "frequency_Hz": float(welch_frequency[peak]),
                    "psd": float(density[peak]),
                    "prominence": float(prominence),
                    "interpretation": "spectral candidate only; not an automatically identified MHD mode",
                })
        out["spectral_peak_candidates"] = peak_rows
        out["spectral_peak_prominence_fraction"] = float(peak_prominence_fraction)
        out["spectral_peak_note"] = "Peak finding is statistical signal processing. Physical mode identity requires equilibrium, diagnostic geometry, channel timing and corroborating diagnostics."
        if compute_cross_channel_spectra and f.size > 1:
            pair_rows = []
            for i in range(f.size):
                for j in range(i + 1, f.size):
                    cross_f, cross_p = cross_spectral_density(
                        y0[:, i], y0[:, j], fs=fs, window="hann",
                        nperseg=welch_nperseg, noverlap=welch_noverlap,
                        detrend=False, scaling="density", return_onesided=True,
                    )
                    coh_f, coh = signal_coherence(
                        y0[:, i], y0[:, j], fs=fs, window="hann",
                        nperseg=welch_nperseg, noverlap=welch_noverlap,
                        detrend=False,
                    )
                    phase = np.unwrap(np.angle(cross_p))
                    coherent = (coh >= 0.5) & (cross_f > 0) & np.isfinite(phase)
                    delay_s = None
                    if np.count_nonzero(coherent) >= 3:
                        slope, intercept = np.polyfit(2.0 * np.pi * cross_f[coherent], phase[coherent], 1)
                        delay_s = float(-slope)
                    pair_rows.append({
                        "channel_i": int(i), "channel_j": int(j),
                        "frequency_Hz": cross_f,
                        "cross_spectrum_real": np.real(cross_p),
                        "cross_spectrum_imag": np.imag(cross_p),
                        "coherence_frequency_Hz": coh_f,
                        "coherence": coh,
                        "cross_phase_rad": phase,
                        "estimated_relative_delay_s": delay_s,
                        "delay_estimation_note": "Phase-slope estimate over bins with coherence >= 0.5; not a hardware synchronization calibration.",
                    })
            out["cross_channel_spectra"] = pair_rows
            out["cross_channel_spectra_note"] = "Cross-spectra/coherence are signal statistics, not automatic MHD-mode identification; channel timing/phase response must be independently known."
        if noise_std is not None:
            n = np.asarray(noise_std, dtype=float)
            if np.any(~np.isfinite(n)) or np.any(n < 0):
                raise ValueError("noise_std must be finite and non-negative.")
            if n.ndim == 1 and n.shape == (1,):
                n = np.asarray(float(n[0]))
            if n.ndim > 1 or (n.ndim == 1 and n.shape != (f.size,)):
                raise ValueError("noise_std must be a scalar, one value, or one value per frequency channel.")
            out["noise_std"] = n
            out["above_noise_rms"] = out["fluctuation_rms"] > n
            noise_per_channel = np.broadcast_to(n, (f.size,))
            expected_noise_psd = np.broadcast_to(
                (noise_per_channel**2 / fs), (welch_frequency.size, f.size)
            ).copy()
            if welch_frequency.size > 1:
                if welch_nperseg % 2 == 0:
                    expected_noise_psd[1:-1, :] *= 2.0
                else:
                    expected_noise_psd[1:, :] *= 2.0
            out["white_noise_psd_expected"] = expected_noise_psd
            out["welch_psd_above_expected_white_noise"] = welch_density > expected_noise_psd
    return out

def reconstruct_ece_temperature_profile(
    frequency_Hz, calibrated_brightness_temperature_eV, time_s, *,
    major_radius_m, B_axis_T, harmonic=2, optical_depth=None,
    background_temperature_eV=0.0, assume_optically_thick=False,
    profile_radius_m=None, time_window_s=None, channel_bandwidth_Hz=None,
    brightness_temperature_uncertainty_eV=None, frequency_uncertainty_Hz=None,
    optical_depth_uncertainty=None, background_temperature_uncertainty_eV=0.0,
    major_radius_uncertainty_m=0.0, B_axis_uncertainty_T=0.0,
    include_time_resolved_profile=False, profile_method="pchip",
    equilibrium_radius_m=None, equilibrium_toroidal_field_T=None,
    channel_resonance_radius_m=None, radius_mapping_uncertainty_m=None,
    plasma_radius_bounds_m=None, channel_covariance_eV2=None,
    temporal_correlation_adjustment=True, electron_energy_eV=None,
    parallel_beta=0.0, model_discrepancy_fraction=0.0,
):
    """Reconstruct a radial Te profile from already-calibrated ECE Tb [eV].

    No detector calibration or signal-to-temperature conversion is performed.
    Resonance radii use only the simplified B_phi(R)=B_axis*R_axis/R model.

    If optical_depth is supplied, Te is inferred from the single-slab LTE
    relation Tb = Te*(1-exp(-tau)) + Tbg*exp(-tau). Otherwise the user must
    explicitly assert optically thick emission, in which case Te ~= Tb.
    The radial profile uses linear or shape-preserving PCHIP interpolation
    between channel locations; there is no extrapolation. A supplied monotonic
    B_phi(R) table or explicit channel resonance positions can replace the
    circular-field approximation, but neither is itself a full ray trace.
    Optional covariance, temporal-correlation, radial-gradient and model-floor
    uncertainty terms are propagated approximately; finite bandwidth remains
    a localization envelope, not a receiver-response convolution.
    """

    f = np.asarray(frequency_Hz, dtype=float)
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(calibrated_brightness_temperature_eV, dtype=float)

    if f.ndim != 1 or f.size < 3 or np.any(~np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("frequency_Hz must contain at least 3 finite, positive, strictly increasing channels.")
    if t.ndim != 1 or t.size < 3 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must contain at least 3 finite, strictly increasing samples.")
    if y.shape != (t.size, f.size) or np.any(~np.isfinite(y)):
        raise ValueError("calibrated brightness temperature must be a finite (time samples, channels) matrix matching time_s and frequency_Hz.")
    if np.any(y < 0):
        raise ValueError("Calibrated brightness temperature samples must be non-negative.")
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)) or harmonic < 1:
        raise ValueError("harmonic must be a positive integer.")
    for name, value, lower, strict in (
        ("major_radius_m", major_radius_m, 0.0, True),
        ("B_axis_T", B_axis_T, 0.0, True),
        ("background_temperature_eV", background_temperature_eV, 0.0, False),
        ("model_discrepancy_fraction", model_discrepancy_fraction, 0.0, False),
    ):
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar) or (scalar <= lower if strict else scalar < lower):
            qualifier = "positive" if strict else "non-negative"
            raise ValueError(f"{name} must be a finite {qualifier} scalar.")
    major_radius_m = float(major_radius_m)
    B_axis_T = float(B_axis_T)
    background_temperature_eV = float(background_temperature_eV)
    model_discrepancy_fraction = float(model_discrepancy_fraction)
    if electron_energy_eV is not None:
        energy_scalar = np.asarray(electron_energy_eV, dtype=float)
        if energy_scalar.ndim != 0 or not np.isfinite(energy_scalar) or energy_scalar < 0:
            raise ValueError("electron_energy_eV must be a finite non-negative scalar.")
        beta_scalar = np.asarray(parallel_beta, dtype=float)
        if beta_scalar.ndim != 0 or not np.isfinite(beta_scalar) or abs(beta_scalar) >= 1:
            raise ValueError("parallel_beta must be a finite scalar strictly between -1 and 1.")
        electron_energy_eV = float(energy_scalar)
        parallel_beta = float(beta_scalar)
    for uncertainty_name, uncertainty_value in (
        ("background_temperature_uncertainty_eV", background_temperature_uncertainty_eV),
        ("major_radius_uncertainty_m", major_radius_uncertainty_m),
        ("B_axis_uncertainty_T", B_axis_uncertainty_T),
    ):
        uncertainty_scalar = np.asarray(uncertainty_value, dtype=float)
        if uncertainty_scalar.ndim != 0 or not np.isfinite(uncertainty_scalar) or uncertainty_scalar < 0:
            raise ValueError(f"{uncertainty_name} must be a finite non-negative scalar.")
    background_temperature_uncertainty_eV = float(background_temperature_uncertainty_eV)
    major_radius_uncertainty_m = float(major_radius_uncertainty_m)
    B_axis_uncertainty_T = float(B_axis_uncertainty_T)
    if not isinstance(include_time_resolved_profile, (bool, np.bool_)):
        raise ValueError("include_time_resolved_profile must be boolean.")
    if not isinstance(profile_method, str) or profile_method not in {"linear", "pchip"}:
        raise ValueError("profile_method must be the string 'linear' or 'pchip'.")
    if not isinstance(temporal_correlation_adjustment, (bool, np.bool_)):
        raise ValueError("temporal_correlation_adjustment must be boolean.")
    if plasma_radius_bounds_m is not None:
        plasma_bounds = np.asarray(plasma_radius_bounds_m, dtype=float)
        if plasma_bounds.shape != (2,) or np.any(~np.isfinite(plasma_bounds)) or plasma_bounds[0] <= 0 or plasma_bounds[0] >= plasma_bounds[1]:
            raise ValueError("plasma_radius_bounds_m must be finite (R_min, R_max) with 0 < R_min < R_max.")
    else:
        plasma_bounds = None
    if (equilibrium_radius_m is None) != (equilibrium_toroidal_field_T is None):
        raise ValueError("equilibrium_radius_m and equilibrium_toroidal_field_T must be supplied together.")
    if channel_resonance_radius_m is not None:
        explicit_radius = np.asarray(channel_resonance_radius_m, dtype=float)
        if explicit_radius.shape != f.shape or np.any(~np.isfinite(explicit_radius)) or np.any(explicit_radius <= 0):
            raise ValueError("channel_resonance_radius_m must contain one finite positive radius per frequency channel.")
        if equilibrium_radius_m is not None:
            raise ValueError("Use either explicit channel_resonance_radius_m or an equilibrium B_phi(R) table, not both.")
    else:
        explicit_radius = None
    if equilibrium_radius_m is not None:
        eq_r = np.asarray(equilibrium_radius_m, dtype=float)
        eq_b = np.asarray(equilibrium_toroidal_field_T, dtype=float)
        if eq_r.ndim != 1 or eq_b.shape != eq_r.shape or eq_r.size < 3 or np.any(~np.isfinite(eq_r)) or np.any(~np.isfinite(eq_b)) or np.any(eq_r <= 0) or np.any(eq_b <= 0) or np.any(np.diff(eq_r) <= 0):
            raise ValueError("Equilibrium radius/field arrays must be finite, positive, strictly increasing and have matching length >= 3.")
    else:
        eq_r = eq_b = None
    if radius_mapping_uncertainty_m is not None:
        radius_mapping_uncertainty_input = np.asarray(radius_mapping_uncertainty_m, dtype=float)
        if radius_mapping_uncertainty_input.ndim > 1 or np.any(~np.isfinite(radius_mapping_uncertainty_input)) or np.any(radius_mapping_uncertainty_input < 0):
            raise ValueError("radius_mapping_uncertainty_m must be finite and non-negative, scalar or per-channel.")
    else:
        radius_mapping_uncertainty_input = None
    if channel_covariance_eV2 is not None:
        covariance = np.asarray(channel_covariance_eV2, dtype=float)
        covariance_scale = max(float(np.max(np.abs(covariance))) if covariance.size else 0.0, np.finfo(float).tiny)
        if (covariance.shape != (f.size, f.size) or np.any(~np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-12 * covariance_scale)):
            raise ValueError("channel_covariance_eV2 must be a finite symmetric channel-by-channel covariance matrix.")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-10 * covariance_scale:
            raise ValueError("channel_covariance_eV2 must be positive semidefinite.")

    def _channel_array(value, name, *, allow_time_matrix=False, strictly_positive=False):
        arr = np.asarray(value, dtype=float)
        if allow_time_matrix and arr.shape == y.shape:
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
            return arr
        if arr.ndim == 0 or arr.shape == (1,):
            arr = np.full(f.shape, float(arr.reshape(-1)[0]))
        elif arr.shape != f.shape:
            expected = "scalar, one value, per-channel, or time-by-channel" if allow_time_matrix else "scalar, one value, or per-channel"
            raise ValueError(f"{name} must be {expected}.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0) or (strictly_positive and np.any(arr <= 0)):
            bound = "finite and strictly positive" if strictly_positive else "finite and non-negative"
            raise ValueError(f"{name} must be {bound}.")
        return arr

    bandwidth = None if channel_bandwidth_Hz is None else _channel_array(
        channel_bandwidth_Hz, "channel_bandwidth_Hz", strictly_positive=True
    )
    if bandwidth is not None and np.any(bandwidth >= 2.0 * f):
        raise ValueError("Each channel bandwidth must be smaller than twice its center frequency.")
    frequency_sigma = None if frequency_uncertainty_Hz is None else _channel_array(
        frequency_uncertainty_Hz, "frequency_uncertainty_Hz"
    )
    if frequency_sigma is not None and np.any(frequency_sigma >= f):
        raise ValueError("frequency_uncertainty_Hz must be smaller than each channel center frequency.")
    tb_sigma_input = None if brightness_temperature_uncertainty_eV is None else _channel_array(
        brightness_temperature_uncertainty_eV, "brightness_temperature_uncertainty_eV",
        allow_time_matrix=True
    )
    tau_sigma = None if optical_depth_uncertainty is None else _channel_array(
        optical_depth_uncertainty, "optical_depth_uncertainty"
    )

    if optical_depth is None:
        if not assume_optically_thick:
            raise ValueError("Supply optical_depth per channel or explicitly assert assume_optically_thick=True.")
        tau = None
    else:
        tau = np.asarray(optical_depth, dtype=float)
        if tau.ndim == 0 or tau.shape == (1,):
            tau = np.full(f.shape, float(tau.reshape(-1)[0]))
        if tau.shape != f.shape or np.any(~np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError("optical_depth must be finite, positive, and scalar or one value per channel.")
        if tau_sigma is not None and tau_sigma.shape != f.shape:
            raise ValueError("optical_depth_uncertainty requires optical_depth to be supplied.")
    if tau is None and tau_sigma is not None:
        raise ValueError("optical_depth_uncertainty requires optical_depth to be supplied.")
    if tau is None and background_temperature_uncertainty_eV > 0:
        raise ValueError("background_temperature_uncertainty_eV is only used when optical_depth is supplied.")

    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), rtol=1e-3, atol=0):
        raise ValueError("time_s must be approximately uniformly sampled; inspect or explicitly resample acquisition data.")

    if time_window_s is None:
        window_start, window_end = float(t[0]), float(t[-1])
    else:
        window = np.asarray(time_window_s, dtype=float)
        if window.shape != (2,) or np.any(~np.isfinite(window)) or window[0] > window[1]:
            raise ValueError("time_window_s must contain finite (start_s, end_s) with start <= end.")
        window_start, window_end = map(float, window)
        if window_start < t[0] or window_end > t[-1]:
            raise ValueError("time_window_s must lie within the supplied time_s range.")
    time_tolerance = 8.0 * np.finfo(float).eps * max(
        abs(window_start), abs(window_end), float(np.median(np.diff(t))), np.finfo(float).tiny
    )
    selected = (t >= window_start - time_tolerance) & (t <= window_end + time_tolerance)
    if np.count_nonzero(selected) < 1:
        raise ValueError("time_window_s contains no acquired samples.")
    selected_time = t[selected]
    selected_signal = y[selected, :]
    mean_signal = np.mean(selected_signal, axis=0)
    tb_eV = mean_signal
    tb_mean_sigma = None
    effective_sample_count = np.full(f.size, float(selected_signal.shape[0]))
    if temporal_correlation_adjustment and selected_signal.shape[0] > 2:
        for channel in range(f.size):
            centered = selected_signal[:, channel] - np.mean(selected_signal[:, channel])
            energy = float(np.dot(centered, centered))
            if energy <= np.finfo(float).tiny:
                continue
            acf = fftconvolve(centered, centered[::-1], mode="full")[centered.size - 1:] / energy
            positive = acf[1:]
            cutoff = np.flatnonzero(positive <= 0)
            stop = int(cutoff[0]) if cutoff.size else positive.size
            tau_int = max(1.0, 1.0 + 2.0 * float(np.sum(positive[:stop])))
            effective_sample_count[channel] = max(1.0, min(float(centered.size), centered.size / tau_int))
    if tb_sigma_input is not None:
        sigma_selected = tb_sigma_input[selected, :] if tb_sigma_input.shape == y.shape else np.broadcast_to(tb_sigma_input, selected_signal.shape)
        tb_mean_sigma = np.sqrt(np.sum(sigma_selected**2, axis=0)) / selected_signal.shape[0]
        if temporal_correlation_adjustment:
            tb_mean_sigma *= np.sqrt(selected_signal.shape[0] / effective_sample_count)
    if channel_covariance_eV2 is not None:
        if tb_mean_sigma is None:
            tb_mean_sigma = np.sqrt(np.maximum(0.0, np.diag(covariance)))
        else:
            tb_mean_sigma = np.sqrt(np.maximum(0.0, tb_mean_sigma**2 + np.diag(covariance)))
    if np.any(~np.isfinite(tb_eV)) or np.any(tb_eV < 0):
        raise ValueError("Calibrated brightness temperature must be finite and non-negative.")
    te_sigma = None
    optical_depth_condition_warning = bool(tau is not None and np.any(tau < 1e-3))
    if tau is not None and tb_mean_sigma is not None:
        sensitivity = 1.0 / (-np.expm1(-tau))
        relative_tb_sigma = tb_mean_sigma / np.maximum(np.abs(tb_eV), np.finfo(float).tiny)
        if np.any(sensitivity * relative_tb_sigma > 0.5):
            optical_depth_condition_warning = True
    optical_depth_condition_message = (
        "At least one optical depth is below 1e-3; the single-slab Te inversion is strongly "
        "ill-conditioned and highly sensitive to Tb, background and tau uncertainty."
        if optical_depth_condition_warning else None
    )
    if tau is None:
        te_eV = tb_eV.copy()
        if tb_mean_sigma is not None:
            te_sigma = tb_mean_sigma.copy()
        inference = "Tb approximates Te under the user-declared optically thick assumption."
    else:
        emissivity = -np.expm1(-tau)
        exp_tau = np.exp(-tau)
        te_eV = (tb_eV - background_temperature_eV * exp_tau) / emissivity
        if np.any(~np.isfinite(te_eV)) or np.any(te_eV < 0):
            raise ValueError("Optical-depth inversion produced non-physical Te; check calibration, background and optical depth.")
        variance = np.zeros(f.shape, dtype=float)
        if tb_mean_sigma is not None:
            variance += (tb_mean_sigma / emissivity)**2
        if tau_sigma is not None:
            dte_dtau = exp_tau * (background_temperature_eV - tb_eV) / emissivity**2
            variance += (dte_dtau * tau_sigma)**2
        if background_temperature_uncertainty_eV:
            dte_dtbg = -exp_tau / emissivity
            variance += (dte_dtbg * background_temperature_uncertainty_eV)**2
        if tb_mean_sigma is not None or tau_sigma is not None or background_temperature_uncertainty_eV:
            te_sigma = np.sqrt(variance)
        inference = "Single-slab LTE inversion using user-supplied optical depth and background temperature."
    if tau is not None and te_sigma is not None:
        relative_te_uncertainty = te_sigma / np.maximum(np.abs(te_eV), np.finfo(float).tiny)
        if np.any(relative_te_uncertainty > 0.5):
            optical_depth_condition_warning = True
    if optical_depth_condition_warning and optical_depth_condition_message is None:
        optical_depth_condition_message = "The supplied uncertainties imply strongly amplified uncertainty in the optical-depth inversion; the 1e-3 threshold is only a heuristic, not a universal validity boundary."

    b_res = (resonance_field_T(f, harmonic) if electron_energy_eV is None else
             resonance_field_relativistic_T(f, harmonic, electron_energy_eV=electron_energy_eV,
                                            parallel_beta=parallel_beta))
    if explicit_radius is not None:
        radius = explicit_radius.copy()
        mapping_model = "user-supplied channel resonance locations; use only if derived from a validated equilibrium/line-of-sight model"
    elif eq_r is None:
        radius = float(major_radius_m) * float(B_axis_T) / b_res
        mapping_model = "simplified circular toroidal field B(R)=B_axis*R_axis/R"
    else:
        if np.any(b_res < np.min(eq_b)) or np.any(b_res > np.max(eq_b)):
            raise ValueError("Resonance field falls outside the supplied equilibrium B_phi(R) table; extrapolation is forbidden.")
        if not (np.all(np.diff(eq_b) > 0) or np.all(np.diff(eq_b) < 0)):
            raise ValueError("Supplied equilibrium B_phi(R) must be strictly monotonic for single-valued inversion.")
        radius = np.interp(b_res, eq_b if eq_b[0] < eq_b[-1] else eq_b[::-1],
                           eq_r if eq_b[0] < eq_b[-1] else eq_r[::-1])
        mapping_model = "user-supplied monotonic equilibrium B_phi(R) table; not a full ray-tracing/flux-coordinate solution"
    if explicit_radius is not None:
        radius_sigma = np.zeros_like(radius)
    elif eq_r is None:
        radius_sigma = np.abs(radius) * np.sqrt(
            (float(major_radius_uncertainty_m) / float(major_radius_m))**2
            + (float(B_axis_uncertainty_T) / float(B_axis_T))**2
            + ((np.zeros_like(f) if frequency_sigma is None else frequency_sigma) / f)**2
        )
    else:
        b_axis_sorted = eq_b if eq_b[0] < eq_b[-1] else eq_b[::-1]
        r_axis_sorted = eq_r if eq_b[0] < eq_b[-1] else eq_r[::-1]
        dr_db = np.gradient(r_axis_sorted, b_axis_sorted)
        local_dr_db = np.interp(b_res, b_axis_sorted, dr_db)
        sigma_b_from_frequency = b_res * ((np.zeros_like(f) if frequency_sigma is None else frequency_sigma) / f)
        radius_sigma = np.abs(local_dr_db * sigma_b_from_frequency)
    bandwidth_radius_lower = bandwidth_radius_upper = None
    if bandwidth is not None and explicit_radius is None:
        if electron_energy_eV is None:
            b_low_frequency = resonance_field_T(f - bandwidth / 2.0, harmonic)
            b_high_frequency = resonance_field_T(f + bandwidth / 2.0, harmonic)
        else:
            b_low_frequency = resonance_field_relativistic_T(f - bandwidth / 2.0, harmonic, electron_energy_eV=electron_energy_eV, parallel_beta=parallel_beta)
            b_high_frequency = resonance_field_relativistic_T(f + bandwidth / 2.0, harmonic, electron_energy_eV=electron_energy_eV, parallel_beta=parallel_beta)
        if eq_r is None:
            bandwidth_radius_lower = float(major_radius_m) * float(B_axis_T) / b_high_frequency
            bandwidth_radius_upper = float(major_radius_m) * float(B_axis_T) / b_low_frequency
        else:
            if np.any(b_low_frequency < np.min(eq_b)) or np.any(b_high_frequency > np.max(eq_b)):
                raise ValueError("Channel bandwidth extends outside the supplied equilibrium B_phi(R) table.")
            b_sorted = eq_b if eq_b[0] < eq_b[-1] else eq_b[::-1]
            r_sorted_eq = eq_r if eq_b[0] < eq_b[-1] else eq_r[::-1]
            radius_at_low_f = np.interp(b_low_frequency, b_sorted, r_sorted_eq)
            radius_at_high_f = np.interp(b_high_frequency, b_sorted, r_sorted_eq)
            bandwidth_radius_lower = np.minimum(radius_at_low_f, radius_at_high_f)
            bandwidth_radius_upper = np.maximum(radius_at_low_f, radius_at_high_f)
    if radius_mapping_uncertainty_input is not None:
        if radius_mapping_uncertainty_input.ndim == 1 and radius_mapping_uncertainty_input.size not in {1, f.size}:
            raise ValueError("radius_mapping_uncertainty_m must be scalar, one value, or contain one value per frequency channel.")
        radius_sigma = np.sqrt(radius_sigma**2 + np.broadcast_to(radius_mapping_uncertainty_input, f.shape)**2)
    if plasma_bounds is not None and np.any((radius < plasma_bounds[0]) | (radius > plasma_bounds[1])):
        raise ValueError("One or more resonance radii fall outside the user-supplied plasma major-radius bounds.")
    order = np.argsort(radius)
    r_sorted = radius[order]
    if np.any(np.diff(r_sorted) <= 0):
        raise ValueError("Channel resonance radii must be distinct to reconstruct a single-valued radial profile.")
    te_sorted = te_eV[order]
    radius_sigma_sorted = radius_sigma[order]
    te_sigma_sorted = None if te_sigma is None else te_sigma[order]
    if np.any(np.diff(r_sorted) <= 0):
        raise ValueError("Resonance radii must be distinct to reconstruct a profile.")
    if profile_radius_m is None:
        r_profile = np.linspace(float(r_sorted[0]), float(r_sorted[-1]), 100)
    else:
        r_profile = np.asarray(profile_radius_m, dtype=float)
        if r_profile.ndim != 1 or r_profile.size < 2 or np.any(~np.isfinite(r_profile)):
            raise ValueError("profile_radius_m must be a finite 1-D grid with at least 2 points.")
        if np.any(np.diff(r_profile) <= 0) or r_profile[0] < r_sorted[0] or r_profile[-1] > r_sorted[-1]:
            raise ValueError("profile_radius_m must be strictly increasing and remain within the measured resonance-radius range.")

    profile_te = PchipInterpolator(r_sorted, te_sorted, extrapolate=False)(r_profile) if profile_method == "pchip" else np.interp(r_profile, r_sorted, te_sorted)
    profile_te_sigma = None
    if te_sigma_sorted is not None:
        right = np.clip(np.searchsorted(r_sorted, r_profile, side="right"), 1, r_sorted.size - 1)
        left = right - 1
        fraction = (r_profile - r_sorted[left]) / (r_sorted[right] - r_sorted[left])
        profile_te_sigma = np.sqrt(((1.0 - fraction) * te_sigma_sorted[left])**2 + (fraction * te_sigma_sorted[right])**2)
    if channel_covariance_eV2 is not None:
        # Propagate the full covariance through the selected interpolation by a
        # numerical Jacobian. This retains common-mode and cross-channel terms.
        cov_sorted = covariance[np.ix_(order, order)]
        if tau is None:
            dte_dtb_sorted = np.ones(f.size)
        else:
            dte_dtb_sorted = (1.0 / (-np.expm1(-tau)))[order]
        te_cov_sorted = cov_sorted * np.outer(dte_dtb_sorted, dte_dtb_sorted)
        jacobian = np.empty((r_profile.size, f.size), dtype=float)
        for channel in range(f.size):
            step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(te_sorted[channel]))
            perturbed = te_sorted.copy()
            perturbed[channel] += step
            if profile_method == "pchip":
                perturbed_profile = PchipInterpolator(r_sorted, perturbed, extrapolate=False)(r_profile)
            else:
                perturbed_profile = np.interp(r_profile, r_sorted, perturbed)
            jacobian[:, channel] = (perturbed_profile - profile_te) / step
        profile_covariance_variance = np.einsum("ij,jk,ik->i", jacobian, te_cov_sorted, jacobian)
        diagonal_covariance_variance = np.einsum(
            "ij,j,ij->i", jacobian, np.diag(te_cov_sorted), jacobian
        )
        residual_independent_variance = (
            np.zeros(r_profile.size) if profile_te_sigma is None else
            np.maximum(0.0, profile_te_sigma**2 - diagonal_covariance_variance)
        )
        profile_te_sigma = np.sqrt(np.maximum(0.0, profile_covariance_variance + residual_independent_variance))
    profile_radius_sigma = np.interp(r_profile, r_sorted, radius_sigma_sorted)
    local_gradient = np.gradient(profile_te, r_profile, edge_order=2)
    profile_te_sigma_from_radius = np.abs(local_gradient) * profile_radius_sigma
    if profile_te_sigma is not None:
        profile_te_sigma = np.sqrt(profile_te_sigma**2 + profile_te_sigma_from_radius**2)
        if model_discrepancy_fraction > 0:
            profile_te_sigma = np.sqrt(profile_te_sigma**2 + (model_discrepancy_fraction * np.abs(profile_te))**2)
    elif model_discrepancy_fraction > 0:
        profile_te_sigma = np.sqrt((model_discrepancy_fraction * np.abs(profile_te))**2 + profile_te_sigma_from_radius**2)
    elif np.any(profile_radius_sigma > 0):
        profile_te_sigma = profile_te_sigma_from_radius.copy()
    output = {
        "frequency_Hz": f,
        "harmonic": int(harmonic),
        "mean_calibrated_brightness_temperature_eV": mean_signal,
        "selected_time_window_s": [window_start, window_end],
        "selected_sample_count": int(selected_time.size),
        "selected_time_s": selected_time,
        "brightness_temperature_eV": tb_eV,
        "brightness_temperature_uncertainty_eV": tb_mean_sigma,
        "effective_sample_count_per_channel": effective_sample_count,
        "temporal_correlation_adjustment_applied": bool(temporal_correlation_adjustment),
        "electron_temperature_channel_eV": te_eV,
        "electron_temperature_channel_uncertainty_eV": te_sigma,
        "resonance_field_T": b_res,
        "resonance_radius_m": radius,
        "resonance_radius_uncertainty_m": radius_sigma,
        "profile_radius_m": r_profile,
        "profile_radius_uncertainty_m": profile_radius_sigma,
        "electron_temperature_profile_eV": profile_te,
        "electron_temperature_profile_uncertainty_eV": profile_te_sigma,
        "channel_bandwidth_Hz": bandwidth,
        "resonance_radius_bandwidth_lower_m": bandwidth_radius_lower,
        "resonance_radius_bandwidth_upper_m": bandwidth_radius_upper,
        "inference_basis": inference,
        "optical_depth_condition_warning": optical_depth_condition_warning,
        "optical_depth_condition_message": optical_depth_condition_message,
        "optical_depth_inversion_sensitivity_dTe_dTb": None if tau is None else 1.0 / (-np.expm1(-tau)),
        "profile_method": f"{profile_method} interpolation between channel resonance locations; no extrapolation; grid points are not independent measurements",
        "profile_gradient_eV_per_m": local_gradient,
        "plasma_radius_bounds_m": None if plasma_bounds is None else plasma_bounds.tolist(),
        "profile_spatial_quality": {
            "measured_channel_count": int(f.size),
            "output_grid_point_count": int(r_profile.size),
            "minimum_channel_spacing_m": float(np.min(np.diff(r_sorted))),
            "maximum_channel_spacing_m": float(np.max(np.diff(r_sorted))),
            "output_grid_is_not_measurement_resolution": True,
            "feature_detection_status": "gradient-based flat-region candidates only; not transport-barrier identification",
        },
        "profile_uncertainty_from_radius_eV": profile_te_sigma_from_radius,
        "profile_interpolation_method_sensitivity": (
            float(np.max(np.abs(np.interp(r_profile, r_sorted, te_sorted) - profile_te)))
            if profile_method == "pchip" else None
        ),
        "equilibrium_model": mapping_model,
        "model_discrepancy_fraction_user_supplied": float(model_discrepancy_fraction),
        "resonance_model": ("cold non-relativistic resonance" if electron_energy_eV is None else "monoenergetic relativistic/Doppler correction; not distribution averaged"),
        "uncertainty_method": "first-order propagation with optional channel covariance, temporal autocorrelation estimate, local radial-gradient contribution, and user-specified fractional model-discrepancy floor",
        "assumptions": [
            "input channels are already calibrated brightness temperatures in eV; no calibration or conversion is performed",
            ("cold non-relativistic resonance mapping" if electron_energy_eV is None else
             "monoenergetic relativistic/Doppler resonance correction; not distribution averaged"),
            "channel center frequencies represent channel response",
            "profile is inferred only between measured channel resonance locations",
        ],
        "limitations": [
            "not EFIT equilibrium or ray tracing",
            ("no relativistic or Doppler correction" if electron_energy_eV is None else
             "monoenergetic relativistic/Doppler correction is approximate; distribution effects and full ray tracing are not modeled"),
            "bandwidth is represented as a radial localization envelope, not a full receiver-response convolution",
            "uncertainties are approximate first-order estimates; include covariance only when supplied and model discrepancy only through the optional user-specified floor",
            "temperature inference depends on user-supplied optical depth or optically thick assertion",
            "time-resolved profiles are optional and are not a substitute for a radiative-transfer inversion",
        ],
    }
    grad_abs = np.abs(local_gradient)
    flat_threshold = max(float(np.nanpercentile(grad_abs, 25)), np.finfo(float).eps)
    output["flat_region_candidate_mask"] = grad_abs <= flat_threshold
    output["flat_region_candidate_note"] = "Gradient-based candidate mask only; does not identify a transport barrier without equilibrium, uncertainty and transport evidence."
    if include_time_resolved_profile:
        if tau is None:
            te_time = selected_signal.copy()
            te_time_sigma = None
            if tb_sigma_input is not None:
                te_time_sigma = (
                    tb_sigma_input[selected, :].copy() if tb_sigma_input.shape == y.shape
                    else np.broadcast_to(tb_sigma_input, selected_signal.shape).copy()
                )
        else:
            emissivity = -np.expm1(-tau)
            exp_tau = np.exp(-tau)
            te_time = (selected_signal - background_temperature_eV * exp_tau[None, :]) / emissivity[None, :]
            if np.any(~np.isfinite(te_time)) or np.any(te_time < 0):
                raise ValueError("Time-resolved optical-depth inversion produced non-physical Te.")
            te_time_sigma = None
            if tb_sigma_input is not None:
                sigma_time = (
                    tb_sigma_input[selected, :].copy() if tb_sigma_input.shape == y.shape
                    else np.broadcast_to(tb_sigma_input, selected_signal.shape).copy()
                )
                te_time_variance = (sigma_time / emissivity[None, :])**2
                if tau_sigma is not None:
                    dte_dtau_time = exp_tau[None, :] * (background_temperature_eV - selected_signal) / emissivity[None, :]**2
                    te_time_variance += (dte_dtau_time * tau_sigma[None, :])**2
                if background_temperature_uncertainty_eV:
                    dte_dtbg_time = -exp_tau / emissivity
                    te_time_variance += (dte_dtbg_time[None, :] * background_temperature_uncertainty_eV)**2
                te_time_sigma = np.sqrt(te_time_variance)
            elif tau_sigma is not None or background_temperature_uncertainty_eV:
                te_time_variance = np.zeros_like(te_time)
                if tau_sigma is not None:
                    dte_dtau_time = exp_tau[None, :] * (background_temperature_eV - selected_signal) / emissivity[None, :]**2
                    te_time_variance += (dte_dtau_time * tau_sigma[None, :])**2
                if background_temperature_uncertainty_eV:
                    dte_dtbg_time = -exp_tau / emissivity
                    te_time_variance += (dte_dtbg_time[None, :] * background_temperature_uncertainty_eV)**2
                te_time_sigma = np.sqrt(te_time_variance)
        if channel_covariance_eV2 is not None:
            covariance_variance = np.diag(covariance)[None, :]
            if tau is not None:
                covariance_variance = covariance_variance / emissivity[None, :]**2
            if te_time_sigma is None:
                te_time_sigma = np.sqrt(np.broadcast_to(covariance_variance, te_time.shape))
            else:
                te_time_sigma = np.sqrt(te_time_sigma**2 + covariance_variance)
        if te_time_sigma is not None and model_discrepancy_fraction > 0:
            te_time_sigma = np.sqrt(te_time_sigma**2 + (model_discrepancy_fraction * np.abs(te_time))**2)
        output["electron_temperature_channel_time_eV"] = te_time
        sorted_te_time = te_time[:, order]
        if profile_method == "pchip":
            profile_time = PchipInterpolator(
                r_sorted, sorted_te_time, axis=1, extrapolate=False
            )(r_profile)
        else:
            profile_time = np.vstack([
                np.interp(r_profile, r_sorted, row) for row in sorted_te_time
            ])
        output["electron_temperature_profile_time_eV"] = profile_time
        if te_time_sigma is not None:
            profile_time_sigma = np.vstack([
                np.interp(r_profile, r_sorted, row[order]) for row in te_time_sigma
            ])
            if channel_covariance_eV2 is not None:
                # Preserve cross-channel covariance in each time-resolved profile.
                # Remove the diagonal covariance contribution already represented
                # by interpolated channel uncertainties before adding the full matrix.
                covariance_diagonal = np.diag(te_cov_sorted)
                for row_index, channel_te in enumerate(sorted_te_time):
                    jacobian_time = np.empty((r_profile.size, f.size), dtype=float)
                    for channel in range(f.size):
                        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(channel_te[channel]))
                        perturbed = channel_te.copy()
                        perturbed[channel] += step
                        if profile_method == "pchip":
                            perturbed_profile = PchipInterpolator(
                                r_sorted, perturbed, extrapolate=False
                            )(r_profile)
                        else:
                            perturbed_profile = np.interp(r_profile, r_sorted, perturbed)
                        jacobian_time[:, channel] = (perturbed_profile - profile_time[row_index]) / step
                    full_variance = np.einsum(
                        "ij,jk,ik->i", jacobian_time, te_cov_sorted, jacobian_time
                    )
                    diagonal_variance = np.einsum(
                        "ij,j,ij->i", jacobian_time, covariance_diagonal, jacobian_time
                    )
                    independent_variance = np.maximum(
                        0.0, profile_time_sigma[row_index]**2 - diagonal_variance
                    )
                    profile_time_sigma[row_index] = np.sqrt(
                        np.maximum(0.0, full_variance + independent_variance)
                    )
            # Add the radial-location contribution using each time slice's local gradient.
            for row_index, row in enumerate(profile_time):
                profile_time_sigma[row_index] = np.sqrt(
                    profile_time_sigma[row_index]**2
                    + (np.abs(np.gradient(row, r_profile, edge_order=2)) * profile_radius_sigma)**2
                )
            output["electron_temperature_channel_time_uncertainty_eV"] = te_time_sigma
            output["electron_temperature_profile_time_uncertainty_eV"] = profile_time_sigma
    return output

