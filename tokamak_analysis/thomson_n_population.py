"""N-population relativistic optical Thomson forward model.

Each population is an isotropic Maxwell-Juttner component with its own
temperature and number-density fraction. Components share the supplied
scattering-vector flow projection, polarization and geometry. Mixture power
is weighted by component differential cross-sections, not temperature averages.
This is a parametric mixture, not arbitrary-EVDF inversion.
"""
from __future__ import annotations
import numpy as np
from .thomson_kinetic_models import C, relativistic_maxwell_juttner_spectrum

def _real_vector(value, name, minimum=1):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size < minimum:
        raise ValueError(f"{name} must be a real one-dimensional vector with at least {minimum} entries")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)): raise ValueError(f"{name} must contain finite values")
    return out

def n_population_relativistic_thomson_spectrum(wavelength_nm, laser_wavelength_nm, scattering_angle_rad, population_temperatures_eV, population_fractions, *, flow_projection_m_s, incident_polarization, radial_order=36, angular_order=18, maximum_recoil_correction_fraction=0.01, include_compton_recoil=False):
    """Return a cross-section-weighted N-component relativistic MJ spectrum.

    Fractions are number-density fractions and must be non-negative and sum
    to one. All components share the scattering-vector flow projection.
    Zero-fraction components are skipped. Absolute density is not inferred.
    """
    wavelength = _real_vector(wavelength_nm, "wavelength_nm", 7)
    temperatures = _real_vector(population_temperatures_eV, "population_temperatures_eV", 2)
    fractions = _real_vector(population_fractions, "population_fractions", 2)
    if temperatures.shape != fractions.shape: raise ValueError("temperature and fraction vectors must have identical lengths")
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0): raise ValueError("wavelength_nm must be positive and strictly increasing")
    if np.any(temperatures <= 0): raise ValueError("population temperatures must be positive")
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, rtol=1e-9, atol=1e-12): raise ValueError("population fractions must be non-negative and sum to one")
    laser, angle, flow = float(laser_wavelength_nm), float(scattering_angle_rad), float(flow_projection_m_s)
    if not np.isfinite(laser) or laser <= 0: raise ValueError("laser wavelength must be positive and finite")
    if not np.isfinite(angle) or not 0 < angle < np.pi: raise ValueError("scattering angle must lie in (0, pi)")
    if not np.isfinite(flow) or abs(flow) >= C: raise ValueError("flow projection must be finite and subluminal")
    if not isinstance(include_compton_recoil, (bool, np.bool_)): raise ValueError("include_compton_recoil must be boolean")
    components, weighted, total = [], [], 0.0
    for i, (temperature, fraction) in enumerate(zip(temperatures, fractions)):
        if fraction == 0:
            components.append({"index": int(i), "temperature_eV": float(temperature), "number_fraction": 0.0, "evaluated": False})
            continue
        result = relativistic_maxwell_juttner_spectrum(wavelength_nm=wavelength, laser_wavelength_nm=laser, scattering_angle_rad=angle, electron_temperature_eV=float(temperature), flow_projection_m_s=flow, incident_polarization=incident_polarization, radial_order=radial_order, angular_order=angular_order, maximum_recoil_correction_fraction=maximum_recoil_correction_fraction, include_compton_recoil=include_compton_recoil)
        cross = float(result["effective_differential_cross_section_m2_sr"])
        if not np.isfinite(cross) or cross <= 0: raise ValueError(f"population {i} has non-positive differential cross-section")
        weight = float(fraction) * cross
        total += weight
        weighted.append(weight * np.asarray(result["spectral_density_per_nm"], dtype=float))
        components.append({"index": int(i), "temperature_eV": float(temperature), "number_fraction": float(fraction), "power_weight_m2_sr": weight, "effective_differential_cross_section_m2_sr": cross, "evaluated": True})
    if not np.isfinite(total) or total <= 0: raise ValueError("mixture has no positive scattered-power contribution")
    spectrum = np.sum(weighted, axis=0) / total
    if spectrum.shape != wavelength.shape or not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0): raise ValueError("mixture produced an invalid spectrum")
    return {"spectral_density_per_nm": spectrum, "effective_differential_cross_section_m2_sr": float(total), "population_temperatures_eV": temperatures.copy(), "population_number_fractions": fractions.copy(), "population_components": components, "population_count": int(temperatures.size), "flow_projection_m_s": flow, "absolute_density_inferred": False, "model": "N-population isotropic relativistic Maxwell-Juttner Thomson mixture", "assumptions": ["Isotropic Maxwell-Juttner components.", "Shared bulk-flow projection along the scattering vector.", "Number fractions weighted by component differential scattered power.", "Not an arbitrary or anisotropic EVDF reconstruction.", "Absolute density is not inferred."]}
