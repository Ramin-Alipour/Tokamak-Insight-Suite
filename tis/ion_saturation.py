"""Ion-saturation-current conversion to particle-flux and model-dependent density estimates."""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def analyze_ion_saturation_current(
    current_A,
    *,
    collection_area_m2: float,
    ion_mass_amu: float,
    electron_temperature_eV: float,
    ion_temperature_eV: float = 0.0,
    charge_state: int = 1,
    sheath_factor: float = 0.61,
    gamma_i: float = 1.0,
) -> dict:
    """Calculate charge-current flux and a sheath-model density estimate."""
    current = np.asarray(current_A, dtype=float)
    if current.size == 0 or not np.all(np.isfinite(current)):
        raise ValueError("current_A must contain finite values.")
    positive = {
        "collection_area_m2": collection_area_m2,
        "ion_mass_amu": ion_mass_amu,
        "electron_temperature_eV": electron_temperature_eV,
        "sheath_factor": sheath_factor,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not np.isfinite(ion_temperature_eV) or ion_temperature_eV < 0:
        raise ValueError("ion_temperature_eV must be finite and nonnegative.")
    if isinstance(charge_state, bool) or not isinstance(charge_state, (int, np.integer)) or charge_state < 1:
        raise ValueError("charge_state must be a positive integer.")
    if not np.isfinite(gamma_i) or gamma_i < 0:
        raise ValueError("gamma_i must be finite and nonnegative.")
    ion_mass = ion_mass_amu * AMU
    cs = float(np.sqrt(E_CHARGE * (charge_state * electron_temperature_eV + gamma_i * ion_temperature_eV) / ion_mass))
    particle_flux = np.abs(current) / (charge_state * E_CHARGE * collection_area_m2)
    density = np.abs(current) / (sheath_factor * charge_state * E_CHARGE * collection_area_m2 * cs)
    return {
        "diagnostic": "Ion saturation current",
        "input_current_A": current,
        "particle_flux_m-2_s-1": particle_flux,
        "ion_sound_speed_m_s": cs,
        "ion_density_estimate_m-3": density,
        "collection_area_m2": float(collection_area_m2),
        "charge_state": int(charge_state),
        "sheath_factor": float(sheath_factor),
        "assumptions_and_limits": [
            "Current-to-flux conversion assumes a known effective collection area and charge state.",
            "Density estimate depends on the supplied sheath factor and ion sound-speed closure.",
            "Magnetization, sheath expansion, probe geometry, secondary emission and RF pickup are not modeled.",
        ],
    }
