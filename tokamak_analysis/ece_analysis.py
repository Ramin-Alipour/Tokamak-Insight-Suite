"""Electron-cyclotron-emission (ECE) channel and profile analysis.

This module deliberately separates resonance mapping from temperature inference:
frequency-to-radius mapping requires an equilibrium B(R) model, while brightness
temperature is only a plasma-temperature estimate in the optically thick,
thermalized regime.
"""
from __future__ import annotations

import numpy as np

from scipy.constants import elementary_charge as E_CHARGE, electron_mass as M_E, speed_of_light as C_LIGHT, Boltzmann as K_B
from scipy.signal import welch as welch_psd, get_window
from scipy.stats import chi2


def resonance_field_T(frequency_Hz, harmonic: int = 1):
    """Magnetic field [T] for non-relativistic ECE resonance at frequency [Hz]."""
    f = np.asarray(frequency_Hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must contain finite positive values.")
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)) or harmonic < 1:
        raise ValueError("harmonic must be a positive integer.")
    result = 2.0 * np.pi * M_E * f / (harmonic * E_CHARGE)
    return float(result) if result.ndim == 0 else result


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
    tb = C_LIGHT**2 * radiance / (2.0 * K_B * freq**2)
    return float(tb) if tb.ndim == 0 else tb


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
                mode="brightness_temperature"):
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
    try:
        psd_confidence_level = float(psd_confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("psd_confidence_level must be finite and strictly between 0 and 1.") from exc
    if not np.isfinite(psd_confidence_level) or not 0.0 < psd_confidence_level < 1.0:
        raise ValueError("psd_confidence_level must be finite and strictly between 0 and 1.")
    out = {"frequency_Hz": f.copy(), "harmonic": int(harmonic),
           "resonance_field_T": resonance_field_T(f, harmonic),
           "assumptions": ["non-relativistic resonance", "cold-plasma frequency mapping"]}
    if channel_bandwidth_Hz is not None:
        bw = np.asarray(channel_bandwidth_Hz, dtype=float)
        if bw.ndim == 0:
            bw = np.full(f.shape, float(bw))
        if bw.shape != f.shape or np.any(~np.isfinite(bw)) or np.any(bw <= 0):
            raise ValueError("channel_bandwidth_Hz must be positive and scalar or match frequency_Hz.")
        out["channel_bandwidth_Hz"] = bw
        out["fractional_bandwidth"] = bw / f
    if major_radius_m is not None or B_axis_T is not None:
        if major_radius_m is None or B_axis_T is None:
            raise ValueError("major_radius_m and B_axis_T must be provided together.")
        out["resonance_radius_m"] = resonance_radius_m(
            f, major_radius_m, B_axis_T, harmonic=harmonic
        )
        out["equilibrium_model"] = "circular toroidal field B(R)=B_axis*R_axis/R"
        out["limitations"] = ["not EFIT mapping; ignores shaping, poloidal field, relativistic and Doppler shifts"]
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
            if tau.ndim > 1 or (tau.ndim == 1 and tau.shape != f.shape):
                raise ValueError("optical_depth must be a scalar or one value per frequency channel.")
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
        welch_nperseg = min(256, t.size)
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
                    "sampling_uniformity_tolerance": 1e-3})
        if noise_std is not None:
            n = np.asarray(noise_std, dtype=float)
            if np.any(~np.isfinite(n)) or np.any(n < 0):
                raise ValueError("noise_std must be finite and non-negative.")
            if n.ndim > 1 or (n.ndim == 1 and n.shape != (f.size,)):
                raise ValueError("noise_std must be a scalar or one value per frequency channel.")
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
    include_time_resolved_profile=False,
):
    """Reconstruct a radial Te profile from already-calibrated ECE Tb [eV].

    No detector calibration or signal-to-temperature conversion is performed.
    Resonance radii use only the simplified B_phi(R)=B_axis*R_axis/R model.

    If optical_depth is supplied, Te is inferred from the single-slab LTE
    relation Tb = Te*(1-exp(-tau)) + Tbg*exp(-tau). Otherwise the user must
    explicitly assert optically thick emission, in which case Te ~= Tb.
    The radial profile is linearly interpolated between channel locations;
    there is no extrapolation, EFIT mapping, ray tracing or receiver-response
    convolution. Optional independent uncertainties are propagated to first
    order, finite bandwidth is represented by a radial localization envelope,
    and time-resolved profiles can be returned on request.
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
    if not np.isfinite(major_radius_m) or major_radius_m <= 0 or not np.isfinite(B_axis_T) or B_axis_T <= 0:
        raise ValueError("major_radius_m and B_axis_T must be finite and positive.")
    if not np.isfinite(background_temperature_eV) or background_temperature_eV < 0:
        raise ValueError("background_temperature_eV must be finite and non-negative.")
    for uncertainty_name, uncertainty_value in (
        ("background_temperature_uncertainty_eV", background_temperature_uncertainty_eV),
        ("major_radius_uncertainty_m", major_radius_uncertainty_m),
        ("B_axis_uncertainty_T", B_axis_uncertainty_T),
    ):
        if not np.isfinite(uncertainty_value) or uncertainty_value < 0:
            raise ValueError(f"{uncertainty_name} must be finite and non-negative.")
    if not isinstance(include_time_resolved_profile, (bool, np.bool_)):
        raise ValueError("include_time_resolved_profile must be boolean.")

    def _channel_array(value, name, *, allow_time_matrix=False, strictly_positive=False):
        arr = np.asarray(value, dtype=float)
        if allow_time_matrix and arr.shape == y.shape:
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
            return arr
        if arr.ndim == 0:
            arr = np.full(f.shape, float(arr))
        elif arr.shape != f.shape:
            expected = "scalar, per-channel, or time-by-channel" if allow_time_matrix else "scalar or per-channel"
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
        if tau.ndim == 0:
            tau = np.full(f.shape, float(tau))
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
    selected = (t >= window_start) & (t <= window_end)
    if np.count_nonzero(selected) < 1:
        raise ValueError("time_window_s contains no acquired samples.")
    selected_time = t[selected]
    selected_signal = y[selected, :]
    mean_signal = np.mean(selected_signal, axis=0)
    tb_eV = mean_signal
    tb_mean_sigma = None
    if tb_sigma_input is not None:
        sigma_selected = tb_sigma_input[selected, :] if tb_sigma_input.shape == y.shape else np.broadcast_to(tb_sigma_input, selected_signal.shape)
        tb_mean_sigma = np.sqrt(np.sum(sigma_selected**2, axis=0)) / selected_signal.shape[0]
    if np.any(~np.isfinite(tb_eV)) or np.any(tb_eV < 0):
        raise ValueError("Calibrated brightness temperature must be finite and non-negative.")
    te_sigma = None
    optical_depth_condition_warning = bool(tau is not None and np.any(tau < 1e-3))
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

    b_res = resonance_field_T(f, harmonic)
    radius = float(major_radius_m) * float(B_axis_T) / b_res
    radius_sigma = np.abs(radius) * np.sqrt(
        (float(major_radius_uncertainty_m) / float(major_radius_m))**2
        + (float(B_axis_uncertainty_T) / float(B_axis_T))**2
        + ((np.zeros_like(f) if frequency_sigma is None else frequency_sigma) / f)**2
    )
    bandwidth_radius_lower = bandwidth_radius_upper = None
    if bandwidth is not None:
        bandwidth_radius_lower = float(major_radius_m) * float(B_axis_T) / resonance_field_T(f + bandwidth / 2.0, harmonic)
        bandwidth_radius_upper = float(major_radius_m) * float(B_axis_T) / resonance_field_T(f - bandwidth / 2.0, harmonic)
    order = np.argsort(radius)
    r_sorted = radius[order]
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

    profile_te = np.interp(r_profile, r_sorted, te_sorted)
    profile_te_sigma = None
    if te_sigma_sorted is not None:
        right = np.clip(np.searchsorted(r_sorted, r_profile, side="right"), 1, r_sorted.size - 1)
        left = right - 1
        fraction = (r_profile - r_sorted[left]) / (r_sorted[right] - r_sorted[left])
        profile_te_sigma = np.sqrt(((1.0 - fraction) * te_sigma_sorted[left])**2 + (fraction * te_sigma_sorted[right])**2)
    profile_radius_sigma = np.interp(r_profile, r_sorted, radius_sigma_sorted)
    output = {
        "frequency_Hz": f,
        "harmonic": int(harmonic),
        "mean_calibrated_brightness_temperature_eV": mean_signal,
        "selected_time_window_s": [window_start, window_end],
        "selected_sample_count": int(selected_time.size),
        "selected_time_s": selected_time,
        "brightness_temperature_eV": tb_eV,
        "brightness_temperature_uncertainty_eV": tb_mean_sigma,
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
        "profile_method": "linear interpolation between channel resonance locations; no extrapolation",
        "equilibrium_model": "simplified circular toroidal field B(R)=B_axis*R_axis/R",
        "uncertainty_method": "first-order independent-input propagation; excludes model discrepancy and correlations",
        "assumptions": [
            "input channels are already calibrated brightness temperatures in eV; no calibration or conversion is performed",
            "non-relativistic cold resonance mapping",
            "channel center frequencies represent channel response",
            "profile is inferred only between measured channel resonance locations",
        ],
        "limitations": [
            "not EFIT equilibrium or ray tracing",
            "no relativistic or Doppler correction",
            "bandwidth is represented as a radial localization envelope, not a full receiver-response convolution",
            "uncertainties are first-order independent-input estimates and exclude model discrepancy/correlations",
            "temperature inference depends on user-supplied optical depth or optically thick assertion",
            "time-resolved profiles are optional and are not a substitute for a radiative-transfer inversion",
        ],
    }
    if include_time_resolved_profile:
        if tau is None:
            te_time = selected_signal.copy()
        else:
            te_time = (selected_signal - background_temperature_eV * np.exp(-tau)[None, :]) / (-np.expm1(-tau))[None, :]
            if np.any(~np.isfinite(te_time)) or np.any(te_time < 0):
                raise ValueError("Time-resolved optical-depth inversion produced non-physical Te.")
        output["electron_temperature_channel_time_eV"] = te_time
        output["electron_temperature_profile_time_eV"] = np.vstack([
            np.interp(r_profile, r_sorted, row[order]) for row in te_time
        ])
    return output

