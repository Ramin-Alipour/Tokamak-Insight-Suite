"""Electron-cyclotron-emission (ECE) channel and profile analysis.

This module deliberately separates resonance mapping from temperature inference:
frequency-to-radius mapping requires an equilibrium B(R) model, while brightness
temperature is only a plasma-temperature estimate in the optically thick,
thermalized regime.
"""
from __future__ import annotations

import numpy as np

from scipy.constants import elementary_charge as E_CHARGE, electron_mass as M_E, speed_of_light as C_LIGHT, Boltzmann as K_B


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
                noise_std=None, optical_depth=None, mode="brightness_temperature"):
    """Analyze calibrated ECE channels and return validated derived quantities.

    signal may be a (time, channel) fluctuation matrix. Temperature inference
    requires calibrated spectral radiance and is reported as brightness
    temperature; interpreting it as Te requires optical thickness/thermalization.
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
        out.update({"time_s": t, "fluctuation_rms": np.sqrt(np.mean(y0**2, axis=0)),
                    "fluctuation_psd_frequency_Hz": freq, "fluctuation_psd": psd,
                    "sampling_frequency_Hz": 1.0 / float(np.median(dt)),
                    "nyquist_frequency_Hz": 0.5 / float(np.median(dt))})
        if noise_std is not None:
            n = np.asarray(noise_std, dtype=float)
            if np.any(~np.isfinite(n)) or np.any(n < 0):
                raise ValueError("noise_std must be finite and non-negative.")
            if n.ndim > 1 or (n.ndim == 1 and n.shape != (f.size,)):
                raise ValueError("noise_std must be a scalar or one value per frequency channel.")
            out["noise_std"] = n
            out["above_noise_rms"] = out["fluctuation_rms"] > n
    return out

def reconstruct_ece_temperature_profile(
    frequency_Hz, calibrated_brightness_temperature_eV, time_s, *,
    major_radius_m, B_axis_T, harmonic=2, optical_depth=None,
    background_temperature_eV=0.0, assume_optically_thick=False,
    profile_radius_m=None, time_window_s=None,
):
    """Reconstruct a radial Te profile from already-calibrated ECE Tb [eV].

    No detector calibration or signal-to-temperature conversion is performed.
    Resonance radii use only the simplified B_phi(R)=B_axis*R_axis/R model.

    If optical_depth is supplied, Te is inferred from the single-slab LTE
    relation Tb = Te*(1-exp(-tau)) + Tbg*exp(-tau). Otherwise the user must
    explicitly assert optically thick emission, in which case Te ~= Tb.
    The radial profile is linearly interpolated between channel locations;
    there is no extrapolation, EFIT mapping, ray tracing or bandwidth inversion.
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
    if np.any(~np.isfinite(tb_eV)) or np.any(tb_eV < 0):
        raise ValueError("Calibrated brightness temperature must be finite and non-negative.")
    if tau is None:
        te_eV = tb_eV.copy()
        inference = "Tb approximates Te under the user-declared optically thick assumption."
    else:
        emissivity = -np.expm1(-tau)
        te_eV = (tb_eV - background_temperature_eV * np.exp(-tau)) / emissivity
        if np.any(~np.isfinite(te_eV)) or np.any(te_eV < 0):
            raise ValueError("Optical-depth inversion produced non-physical Te; check calibration, background and optical depth.")
        inference = "Single-slab LTE inversion using user-supplied optical depth and background temperature."

    b_res = resonance_field_T(f, harmonic)
    radius = float(major_radius_m) * float(B_axis_T) / b_res
    order = np.argsort(radius)
    r_sorted = radius[order]
    te_sorted = te_eV[order]
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

    return {
        "frequency_Hz": f,
        "harmonic": int(harmonic),
        "mean_calibrated_brightness_temperature_eV": mean_signal,
        "selected_time_window_s": [window_start, window_end],
        "selected_sample_count": int(selected_time.size),
        "brightness_temperature_eV": tb_eV,
        "electron_temperature_channel_eV": te_eV,
        "resonance_field_T": b_res,
        "resonance_radius_m": radius,
        "profile_radius_m": r_profile,
        "electron_temperature_profile_eV": np.interp(r_profile, r_sorted, te_sorted),
        "inference_basis": inference,
        "profile_method": "linear interpolation between channel resonance locations; no extrapolation",
        "equilibrium_model": "simplified circular toroidal field B(R)=B_axis*R_axis/R",
        "assumptions": [
            "input channels are already calibrated brightness temperatures in eV; no calibration or conversion is performed",
            "non-relativistic cold resonance mapping",
            "channel center frequencies represent channel response",
            "profile is inferred only between measured channel resonance locations",
        ],
        "limitations": [
            "not EFIT equilibrium or ray tracing",
            "no relativistic or Doppler correction",
            "no finite-bandwidth convolution or regularized inverse problem",
            "no uncertainty propagation",
            "temperature inference depends on user-supplied optical depth or optically thick assertion",
            "channel values are averaged over the selected time window; time evolution is not reconstructed",
        ],
    }

