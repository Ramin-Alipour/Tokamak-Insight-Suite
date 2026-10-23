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

    Returns the resonance major radius. This is an explicitly simplified
    equilibrium model, not a substitute for EFIT mapping in shaped plasmas.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    r0 = np.asarray(major_radius_m, dtype=float)
    b0 = np.asarray(B_axis_T, dtype=float)
    if np.any(~np.isfinite(r0)) or np.any(r0 <= 0):
        raise ValueError("major_radius_m must be finite and positive.")
    if np.any(~np.isfinite(b0)) or np.any(b0 <= 0):
        raise ValueError("B_axis_T must be finite and positive.")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 (high-field side) or +1 (low-field side).")
    b_res = resonance_field_T(f, harmonic)
    radius = r0 * b0 / b_res
    radius = radius if sign == 1 else 2.0 * r0 - radius
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
    if mode not in {"brightness_temperature", "channel_fluctuations", "resonance_mapping"}:
        raise ValueError("Unsupported mode.")
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
        if rad.shape[-1:] != f.shape:
            raise ValueError("Last radiance dimension must match frequency_Hz.")
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
