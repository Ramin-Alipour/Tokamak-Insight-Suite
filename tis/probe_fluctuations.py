"""Density-fluctuation proxies from electrostatic-probe time traces."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def analyze_probe_density_fluctuations(
    ion_saturation_current_A,
    *,
    electron_temperature_eV=None,
    density_per_current_m3_A: float | None = None,
    sampling_frequency_Hz: float | None = None,
    nperseg: int | None = None,
) -> dict:
    """Estimate relative density fluctuations from Isat with optional Te correction.

    For a Bohm-like collection model Isat ∝ n*sqrt(Te + gamma_i*Ti), a
    first-order proxy is n~/n ≈ Isat~/Isat_bar - 0.5 Te~/Te_bar when Ti effects
    are negligible or absorbed into the approximation. Absolute density is
    returned only with an external density-per-current calibration factor.
    """
    current = np.asarray(ion_saturation_current_A, dtype=float).ravel()
    if current.size < 4 or not np.all(np.isfinite(current)):
        raise ValueError("ion_saturation_current_A must contain at least four finite samples.")
    mean_i = float(np.mean(current))
    if abs(mean_i) <= np.finfo(float).eps:
        raise ValueError("Mean ion-saturation current is too close to zero for relative fluctuations.")
    current_rel = (current - mean_i) / mean_i
    temperature_rel = None
    density_rel = current_rel.copy()
    if electron_temperature_eV is not None:
        te = np.asarray(electron_temperature_eV, dtype=float).ravel()
        if te.size != current.size or not np.all(np.isfinite(te)) or np.any(te <= 0):
            raise ValueError("electron_temperature_eV must be positive, finite and match the current trace.")
        mean_te = float(np.mean(te))
        temperature_rel = (te - mean_te) / mean_te
        density_rel = current_rel - 0.5 * temperature_rel
    if density_per_current_m3_A is not None and (
        not np.isfinite(density_per_current_m3_A) or density_per_current_m3_A <= 0
    ):
        raise ValueError("density_per_current_m3_A must be positive and finite.")
    absolute_density = (
        current * density_per_current_m3_A if density_per_current_m3_A is not None else None
    )
    psd = None
    frequency = None
    if sampling_frequency_Hz is not None:
        if not np.isfinite(sampling_frequency_Hz) or sampling_frequency_Hz <= 0:
            raise ValueError("sampling_frequency_Hz must be positive and finite.")
        n = min(current.size, int(nperseg) if nperseg is not None else min(1024, current.size))
        if n < 4:
            raise ValueError("nperseg must allow at least four samples.")
        if nperseg is not None and (isinstance(nperseg, bool) or not isinstance(nperseg, int) or nperseg < 4):
            raise ValueError("nperseg must be an integer of at least four.")
        frequency, psd = welch(density_rel, fs=sampling_frequency_Hz, nperseg=n)
    return {
        "diagnostic": "Probe-derived density fluctuations",
        "mean_ion_saturation_current_A": mean_i,
        "relative_current_fluctuation": current_rel,
        "relative_electron_temperature_fluctuation": temperature_rel,
        "relative_density_fluctuation_proxy": density_rel,
        "relative_density_rms": float(np.sqrt(np.mean(density_rel**2))),
        "absolute_density_proxy_m-3": absolute_density,
        "frequency_Hz": frequency,
        "relative_density_fluctuation_psd": psd,
        "assumptions_and_limits": [
            "Probe-current fluctuations also contain potential, temperature, sheath and instrumental contributions.",
            "The optional temperature correction is first-order and assumes a Bohm-like collection model.",
            "Absolute density requires an externally established calibration and probe-specific model.",
        ],
    }
