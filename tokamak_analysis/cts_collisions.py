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
    """Validate a physical scalar before coercion; reject bool/string/complex."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a positive finite real scalar")
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return x


def _real_vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty real numeric vector")
    result = raw.astype(float, copy=False)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


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
    ni = _real_vector(ion_densities_m3, "ion_densities_m3")
    z = _real_vector(ion_charge_states, "ion_charge_states")
    masses = _real_vector(ion_masses_amu, "ion_masses_amu")
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



def multispecies_thermal_coulomb_collision_matrix(
    species,
    *,
    coulomb_logarithm_matrix=None,
):
    """Return leading-log thermal test-particle collision frequencies nu_ab.

    Each species mapping supplies name, density_m3, mass_kg, charge_C and
    temperature_eV. Row a is the test population and column b the field
    population. The Maxwellian thermal-speed estimate is

      nu_ab = (4 sqrt(2 pi)/3) n_b (q_a q_b)^2 ln(Lambda_ab) /
              ((4 pi eps0)^2 m_a^2 (vTa^2 + vTb^2)^(3/2)),

    with vTs^2 = kB Ts/ms. Pair impact parameters use the reduced mass and
    relative thermal speed; Debye screening includes every supplied species.

    This is a thermal-rate matrix, not a linearized multispecies Landau
    operator: it has no field-particle terms, conservation projection or
    matching FDT noise and must not be used as an ad-hoc resonance width.
    """
    from collections.abc import Mapping, Sequence

    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)) or not species:
        raise ValueError("species must be a non-empty sequence of species mappings")
    parsed = []
    for index, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"species[{index}] must be a mapping")
        name = str(item.get("name", f"species_{index}"))
        density = _positive(item.get("density_m3"), f"species[{index}].density_m3")
        mass = _positive(item.get("mass_kg"), f"species[{index}].mass_kg")
        temperature = _positive(item.get("temperature_eV"), f"species[{index}].temperature_eV") * E_CHARGE
        charge = item.get("charge_C")
        if isinstance(charge, (bool, np.bool_)) or not isinstance(
            charge, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"species[{index}].charge_C must be a finite non-zero real scalar")
        charge = float(charge)
        if not np.isfinite(charge) or charge == 0.0:
            raise ValueError(f"species[{index}].charge_C must be a finite non-zero real scalar")
        parsed.append((name, density, mass, charge, temperature))

    n_species = len(parsed)
    debye_inverse_squared = sum(
        density * charge**2 / (EPS0 * temperature)
        for _, density, _, charge, temperature in parsed
    )
    debye_length = float(1.0 / np.sqrt(debye_inverse_squared))
    rates = np.zeros((n_species, n_species), dtype=float)
    logs = np.zeros_like(rates)
    impact_classical = np.zeros_like(rates)
    impact_quantum = np.zeros_like(rates)
    relative_thermal_speed = np.zeros_like(rates)
    prefactor = 4.0 * np.sqrt(2.0 * np.pi) / 3.0

    if coulomb_logarithm_matrix is not None:
        raw_logs = np.asarray(coulomb_logarithm_matrix)
        if raw_logs.dtype.kind not in "iuf" or raw_logs.shape != (n_species, n_species):
            raise ValueError("coulomb_logarithm_matrix must be a real (n_species,n_species) matrix")
        logs_input = raw_logs.astype(float, copy=False)
        if np.any(~np.isfinite(logs_input)) or np.any(logs_input < 0.0):
            raise ValueError("coulomb_logarithm_matrix must be finite and non-negative")
        if not np.allclose(logs_input, logs_input.T, rtol=1e-10, atol=1e-12):
            raise ValueError("coulomb_logarithm_matrix must be symmetric")
    else:
        logs_input = None

    for a, (_, _, mass_a, charge_a, temp_a) in enumerate(parsed):
        for b, (_, density_b, mass_b, charge_b, temp_b) in enumerate(parsed):
            vrel2 = temp_a / mass_a + temp_b / mass_b
            vrel = np.sqrt(vrel2)
            reduced_mass = mass_a * mass_b / (mass_a + mass_b)
            b_classical = abs(charge_a * charge_b) / (
                4.0 * np.pi * EPS0 * reduced_mass * vrel2
            )
            b_quantum = H_BAR / (2.0 * reduced_mass * vrel)
            b_min = max(b_classical, b_quantum)
            log_lambda = (
                float(logs_input[a, b]) if logs_input is not None
                else 0.5 * np.log1p((debye_length / b_min)**2)
            )
            rates[a, b] = (
                prefactor * density_b * (charge_a * charge_b)**2 * log_lambda
                / ((4.0 * np.pi * EPS0)**2 * mass_a**2 * vrel**3)
            )
            logs[a, b] = log_lambda
            impact_classical[a, b] = b_classical
            impact_quantum[a, b] = b_quantum
            relative_thermal_speed[a, b] = vrel

    return {
        "species_order": [item[0] for item in parsed],
        "collision_frequency_matrix_s_inverse": rates,
        "coulomb_logarithm_matrix": logs,
        "debye_length_m": debye_length,
        "classical_minimum_impact_parameter_matrix_m": impact_classical,
        "quantum_minimum_impact_parameter_matrix_m": impact_quantum,
        "relative_thermal_speed_matrix_m_s": relative_thermal_speed,
        "row_definition": "test_species_a",
        "column_definition": "field_species_b",
        "collision_rate_model": "Maxwellian_thermal_Rutherford_small_angle_leading_log",
        "collision_operator_status": "thermal_rates_only_no_field_particle_conservation_or_FDT",
        "limitations": [
            "Leading-log thermal rate estimate; not velocity-resolved.",
            "No magnetization, relativistic, strong-coupling or quantum-degenerate corrections.",
            "No conserving Landau operator or matching fluctuation source is constructed.",
            "Do not use as an arbitrary CTS resonance broadening parameter.",
        ],
    }
