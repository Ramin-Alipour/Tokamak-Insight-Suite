"""Classical Coulomb collision rates for microwave CTS validity and diagnostics.

These rates are physical collisionality estimates, not a replacement for a
linearized Landau collision operator in the Full-EM Vlasov-Maxwell solver.
They must not be inserted as an arbitrary resonance width.  The returned
metadata makes that distinction explicit.
"""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
K_B = 1.380649e-23
H_BAR = 1.054571817e-34
M_E = 9.1093837015e-31
AMU = 1.66053906660e-27


def _positive(value, name):
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def coulomb_logarithm_electron(
    electron_density_m3: float,
    electron_temperature_eV: float,
) -> dict:
    """Estimate ln(Lambda) from Debye length and classical/quantum b_min.

    Uses b_min=max(e^2/(4*pi*eps0*kT), hbar/sqrt(2*m_e*kT)) and
    lnLambda=0.5*ln(1+(lambda_D/b_min)^2). This smooth expression remains
    finite near weak-coupling breakdown; a small lnLambda is flagged rather
    than silently clipped to a conventional value.
    """
    ne = _positive(electron_density_m3, "electron_density_m3")
    te = _positive(electron_temperature_eV, "electron_temperature_eV")
    kt = te * E_CHARGE
    debye = np.sqrt(EPS0 * kt / (ne * E_CHARGE**2))
    b_classical = E_CHARGE**2 / (4.0 * np.pi * EPS0 * kt)
    b_quantum = H_BAR / np.sqrt(2.0 * M_E * kt)
    b_min = max(b_classical, b_quantum)
    log_lambda = 0.5 * np.log1p((debye / b_min) ** 2)
    return {
        "debye_length_m": float(debye),
        "minimum_impact_parameter_m": float(b_min),
        "classical_impact_parameter_m": float(b_classical),
        "quantum_impact_parameter_m": float(b_quantum),
        "coulomb_logarithm": float(log_lambda),
        "weakly_coupled_screening_valid": bool(log_lambda >= 2.0),
    }


def full_em_cts_coulomb_collision_rates(
    electron_density_m3: float,
    electron_temperature_eV: float,
    ion_densities_m3,
    ion_charge_states,
    ion_masses_amu,
) -> dict:
    """Return Spitzer small-angle electron collision rates for supplied ions.

    The electron-ion rate is evaluated species by species and summed:
      nu_ei = 4*sqrt(2*pi)/3 * n_i Z_i^2 e^4 lnLambda /
              ((4*pi*eps0)^2 sqrt(m_e) (kT_e)^(3/2)).
    The electron-electron momentum-relaxation scale is reported separately
    with the same classical prefactor and n_e. These are thermal rate scales;
    they do not represent velocity-dependent test-particle or field-particle
    terms of the full Landau operator.
    """
    ne = _positive(electron_density_m3, "electron_density_m3")
    te = _positive(electron_temperature_eV, "electron_temperature_eV")
    ni = np.asarray(ion_densities_m3, dtype=float)
    z = np.asarray(ion_charge_states, dtype=float)
    masses = np.asarray(ion_masses_amu, dtype=float)
    if ni.ndim != 1 or ni.size == 0 or z.shape != ni.shape or masses.shape != ni.shape:
        raise ValueError("ion_densities_m3, ion_charge_states and ion_masses_amu must be matching non-empty 1-D arrays")
    if np.any(~np.isfinite(ni)) or np.any(ni < 0.0):
        raise ValueError("ion_densities_m3 must be finite and non-negative")
    if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("ion_charge_states must be positive and finite")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("ion_masses_amu must be positive and finite")
    screening = coulomb_logarithm_electron(ne, te)
    ln_lambda = screening["coulomb_logarithm"]
    kt = te * E_CHARGE
    prefactor = (
        4.0 * np.sqrt(2.0 * np.pi) / 3.0
        * E_CHARGE**4 * ln_lambda
        / ((4.0 * np.pi * EPS0)**2 * np.sqrt(M_E) * kt**1.5)
    )
    nu_ei_species = prefactor * ni * z**2
    nu_ei = float(np.sum(nu_ei_species))
    nu_ee_scale = float(prefactor * ne)
    z_eff = float(np.sum(ni * z**2) / ne)
    return {
        **screening,
        "electron_ion_collision_frequency_s_inverse": nu_ei,
        "electron_ion_collision_frequency_by_species_s_inverse": nu_ei_species.copy(),
        "electron_electron_collision_frequency_scale_s_inverse": nu_ee_scale,
        "effective_charge_Zeff": z_eff,
        "collision_rate_model": "classical_Spitzer_small_angle_Coulomb_rates",
        "collision_operator_status": "rates_only_full_linearized_Landau_operator_not_implemented",
        "validity": {
            "weakly_coupled_plasma": screening["weakly_coupled_screening_valid"],
            "nonrelativistic_electrons": bool(te * E_CHARGE / (M_E * 299792458.0**2) < 0.1),
            "thermal_rate_only": True,
        },
        "integration_warning": (
            "Do not add these rates to numerical resonance regularization. "
            "A collisional CTS spectrum requires a conserving linearized "
            "collision operator and its consistent fluctuation source/FDT."
        ),
    }
