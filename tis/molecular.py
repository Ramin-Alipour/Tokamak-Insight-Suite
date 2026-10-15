"""Molecular rotational/vibrational excitation-temperature analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def analyze_molecular_boltzmann_plot(
    upper_energy_eV,
    integrated_intensity,
    wavelength_nm,
    transition_probability_s,
    statistical_weight,
) -> dict:
    """Estimate a Boltzmann excitation temperature from resolved molecular lines.

    Inputs must refer to transitions from a consistent molecular band/system and
    use mutually comparable, background-subtracted integrated intensities.
    The fitted temperature is an excitation temperature, not automatically Te
    or Ti. Line-of-sight integration, opacity and non-LTE populations can bias it.
    """
    arrays = [
        np.asarray(upper_energy_eV, dtype=float).ravel(),
        np.asarray(integrated_intensity, dtype=float).ravel(),
        np.asarray(wavelength_nm, dtype=float).ravel(),
        np.asarray(transition_probability_s, dtype=float).ravel(),
        np.asarray(statistical_weight, dtype=float).ravel(),
    ]
    if len({a.size for a in arrays}) != 1 or arrays[0].size < 3:
        raise ValueError("Provide at least three paired transitions in equal-length arrays.")
    energy, intensity, wavelength, probability, weight = arrays
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("All transition inputs must be finite.")
    if np.any(energy < 0) or np.any(intensity <= 0) or np.any(wavelength <= 0):
        raise ValueError("Upper energies must be nonnegative; intensities and wavelengths positive.")
    if np.any(probability <= 0) or np.any(weight <= 0):
        raise ValueError("Transition probabilities and statistical weights must be positive.")
    if np.unique(energy).size < 3:
        raise ValueError("At least three distinct upper-state energies are required.")

    # Photon energy is hc/lambda; converting measured photon intensity to upper
    # state population introduces the wavelength factor before division by A*g.
    population_proxy = intensity * wavelength / (probability * weight)
    log_population = np.log(population_proxy)
    fit = linregress(energy, log_population)
    if not np.isfinite(fit.slope) or fit.slope >= 0:
        raise ValueError("Boltzmann plot has no physically positive excitation temperature (slope must be negative).")
    temperature_eV = float(-1.0 / fit.slope)
    n = int(energy.size)
    residual = log_population - (fit.intercept + fit.slope * energy)
    dof = n - 2
    slope_se = (
        float(np.sqrt(np.sum(residual**2) / dof / np.sum((energy - np.mean(energy))**2)))
        if dof > 0 else None
    )
    return {
        "diagnostic": "Molecular spectroscopy",
        "excitation_temperature_eV": temperature_eV,
        "excitation_temperature_standard_error_eV": (
            float(slope_se / fit.slope**2) if slope_se is not None else None
        ),
        "boltzmann_slope_per_eV": float(fit.slope),
        "boltzmann_intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue**2),
        "n_transitions": n,
        "transition_population_proxy": population_proxy,
        "log_population_proxy": log_population,
        "residuals_log_space": residual,
        "assumptions_and_limits": [
            "Resolved, optically thin transitions with comparable absolute or relative calibration.",
            "Transition probabilities and statistical weights must match the chosen molecular state/branch.",
            "The fitted value is a rotational/vibrational excitation temperature for the selected transitions, not automatically electron or ion temperature.",
            "Non-LTE populations, blends, opacity, background subtraction and line-of-sight gradients can invalidate a single Boltzmann temperature.",
        ],
    }
