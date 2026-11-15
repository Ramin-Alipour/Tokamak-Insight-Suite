"""Shared numerical initializers for electrostatic and full-EM CTS inversions.

The seed estimates in this module are optimizer starting values only.  In
particular, the alpha=1 density seed is not a measured or inferred density.
Keeping this helper in the shared physics layer prevents the Full-EM solver
from importing a legacy diagnostic module merely to initialize an optimizer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ELEMENTARY_CHARGE_C = 1.602176634e-19
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
ELECTRON_MASS_KG = 9.1093837015e-31
ATOMIC_MASS_KG = 1.66053906660e-27
LIGHT_SPEED_M_S = 299792458.0


def _real_vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty real numeric vector")
    result = raw.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a positive finite real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return result


def _scattering_wave_number(geometry, probe_frequency_hz):
    if not isinstance(geometry, Mapping):
        raise ValueError("scattering_geometry must be a mapping")
    if "frequency_resolved_geometry" in geometry:
        raise ValueError(
            "Automatic width-based optimizer seeds require one fixed scattering vector. "
            "For frequency_resolved_geometry, supply electron_density_initial_m3, "
            "electron_temperature_initial_eV and ion_temperature_initial_eV explicitly."
        )
    if "incident_direction" in geometry and "scattered_direction" in geometry:
        incident = _real_vector(geometry["incident_direction"], "incident_direction")
        scattered = _real_vector(geometry["scattered_direction"], "scattered_direction")
        if incident.size != 3 or scattered.size != 3:
            raise ValueError("beam directions must be nonzero 3-vectors")
        incident_norm = float(np.linalg.norm(incident))
        scattered_norm = float(np.linalg.norm(scattered))
        if incident_norm <= 0.0 or scattered_norm <= 0.0:
            raise ValueError("beam directions must be nonzero 3-vectors")
        cosine = float(np.clip((incident / incident_norm) @ (scattered / scattered_norm), -1.0, 1.0))
        angle = float(np.arccos(cosine))
    else:
        angle = _positive_scalar(geometry.get("scattering_angle_rad"), "scattering_angle_rad")
        if angle > np.pi:
            raise ValueError("scattering_angle_rad must lie in (0, pi]")
    if "scattering_angle_rad" in geometry and not np.isclose(
        float(geometry["scattering_angle_rad"]), angle, rtol=1e-5, atol=1e-7
    ):
        raise ValueError("scattering angle conflicts with beam direction vectors")
    return 4.0 * np.pi * probe_frequency_hz / LIGHT_SPEED_M_S * np.sin(angle / 2.0)


def _validated_ion_species(ion_species, ion_fractions):
    if not isinstance(ion_species, Sequence) or isinstance(ion_species, (str, bytes)) or not ion_species:
        raise ValueError("ion_species must be a non-empty sequence")
    fractions = _real_vector(ion_fractions, "ion_fractions")
    if (
        fractions.size != len(ion_species)
        or np.any(fractions <= 0.0)
        or not np.isclose(float(np.sum(fractions)), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError("ion_fractions must be positive charge-density fractions summing to one")
    masses = []
    weights = []
    for index, species in enumerate(ion_species):
        if not isinstance(species, Mapping):
            raise ValueError("each ion species must be a mapping")
        charge = _positive_scalar(species.get("charge_state"), f"ion_species[{index}].charge_state")
        if not charge.is_integer():
            raise ValueError("ion charge_state must be a positive integer")
        mass_amu = _positive_scalar(species.get("mass_amu"), f"ion_species[{index}].mass_amu")
        masses.append(mass_amu * ATOMIC_MASS_KG)
        weights.append(charge * fractions[index])
    return np.asarray(masses, dtype=float), np.asarray(weights, dtype=float)


def estimate_cts_optimizer_initial_parameters(
    frequency_shift_Hz,
    spectrum,
    *,
    probe_frequency_Hz,
    scattering_geometry,
    ion_species,
    ion_fractions,
):
    """Estimate CTS optimizer seeds from the measured spectral shape.

    The electron temperature seed uses the full-spectrum RMS frequency width.
    The ion temperature seed uses the central 68%-weight spectral core.  The
    density seed corresponds to alpha=1 and is only a numerical starting
    point; no returned value is an inference or a physical prior.
    """
    frequency = _real_vector(frequency_shift_Hz, "frequency_shift_Hz")
    power = _real_vector(spectrum, "spectrum")
    if frequency.size != power.size or frequency.size < 12 or np.any(np.diff(frequency) <= 0.0):
        raise ValueError(
            "CTS seed estimation requires matching, increasing frequency and spectrum arrays with >=12 samples"
        )
    if np.any(power < 0.0):
        raise ValueError("calibrated CTS spectrum must be non-negative")
    baseline = float(np.quantile(power, 0.05))
    weights = np.maximum(power - baseline, 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        raise ValueError(
            "CTS spectral-shape seed estimation failed: spectrum has no positive contrast above its baseline"
        )
    centroid = float(np.sum(weights * frequency) / total)
    offsets = frequency - centroid
    variance = float(np.sum(weights * offsets**2) / total)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "CTS spectral-shape seed estimation failed: spectrum has no resolvable frequency width"
        )
    probe_frequency = _positive_scalar(probe_frequency_Hz, "probe_frequency_Hz")
    wave_number = _scattering_wave_number(scattering_geometry, probe_frequency)
    electron_velocity_rms = 2.0 * np.pi * np.sqrt(variance) / wave_number
    electron_temperature = ELECTRON_MASS_KG * electron_velocity_rms**2 / ELEMENTARY_CHARGE_C

    order = np.argsort(np.abs(offsets))
    cumulative = np.cumsum(weights[order]) / total
    selected = order[cumulative <= 0.68]
    if selected.size < 3:
        selected = order[: min(max(3, int(np.ceil(0.25 * frequency.size))), frequency.size)]
    core_weights = weights[selected]
    core_total = float(np.sum(core_weights))
    core_variance = float(np.sum(core_weights * offsets[selected]**2) / core_total)

    masses, spectral_weights = _validated_ion_species(ion_species, ion_fractions)
    effective_ion_mass = float(np.sum(spectral_weights * masses) / np.sum(spectral_weights))
    ion_velocity_rms = 2.0 * np.pi * np.sqrt(max(core_variance, variance * 1e-6)) / wave_number
    ion_temperature = effective_ion_mass * ion_velocity_rms**2 / ELEMENTARY_CHARGE_C

    # alpha=1 is a dimensionless optimizer seed, never an inferred density.
    density_seed = (
        VACUUM_PERMITTIVITY_F_M
        * (electron_temperature * ELEMENTARY_CHARGE_C)
        * wave_number**2
        / ELEMENTARY_CHARGE_C**2
    )
    floor = np.finfo(float).tiny**0.25
    values = {
        "electron_density_m3": max(float(density_seed), floor),
        "electron_temperature_eV": max(float(electron_temperature), floor),
        "ion_temperature_eV": max(float(ion_temperature), floor),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("CTS spectral-shape seed estimation produced non-finite or non-positive optimizer seeds")
    return {
        **values,
        "method": "calibrated_spectrum_RMS_and_central_68_percent_width; alpha=1 density seed",
        "role": "optimizer_initial_estimates_only",
        "density_seed_assumption": "dimensionless collective parameter alpha=1 used only to construct a numerical starting value",
        "spectral_centroid_Hz": centroid,
        "spectral_rms_width_Hz": float(np.sqrt(variance)),
        "spectral_core_rms_width_Hz": float(np.sqrt(core_variance)),
        "scattering_wave_number_rad_m": float(wave_number),
        "effective_ion_mass_amu": float(effective_ion_mass / ATOMIC_MASS_KG),
    }
