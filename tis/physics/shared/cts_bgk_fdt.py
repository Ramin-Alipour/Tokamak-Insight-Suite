"""Self-consistent electrostatic CTS closure for the conserving multispecies BGK kernel.

The kinetic response, Poisson screening and classical matrix FDT are evaluated
with one collision model and one equilibrium. This is an electrostatic
longitudinal reference branch, not a magnetized Full-EM scattering vertex.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.constants import epsilon_0

from .cts_collisional_bgk import conserving_bgk_longitudinal_susceptibility
from .cts_fdt import classical_fdt_dynamic_structure_factor


def analyze_conserving_bgk_cts(
    angular_frequency_rad_s,
    *,
    wave_number_m_inverse: float,
    species: Sequence[Mapping],
    common_temperature_J: float,
    collision_frequencies_s_inverse,
    bulk_flow_m_s: float = 0.0,
    quadrature_order: int = 96,
    quadrature_max_order: int = 384,
    quadrature_convergence_tolerance: float = 1e-5,
    psd_relative_tolerance: float = 1e-8,
) -> dict:
    r"""Return screened multispecies S_ab(k,omega) from conserving BGK + FDT.

    The bare response is R_ab=delta(n_a)/delta(U_b), with U_b an externally
    applied species potential energy. Self-consistent electrostatic screening
    uses V_ab=q_a*q_b/(epsilon_0*k**2) and
    chi=(I-R V)^(-1) R. The classical equilibrium matrix FDT then gives
    S=-(2*T/omega) Im_H[chi] for exp(-i omega t).

    Inputs are SI. The species sequence must match the densities and masses
    passed to the BGK kernel exactly. Zero frequency is rejected because the
    static spectrum requires a separately converged low-frequency limit.
    """
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)) or not species:
        raise ValueError("species must be a non-empty sequence of species mappings")
    density = []
    mass = []
    charge = []
    for index, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"species[{index}] must be a mapping")
        try:
            values = np.asarray(
                [item["density_m3"], item["mass_kg"], item["charge_C"]], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"species[{index}] requires density_m3, mass_kg and charge_C"
            ) from exc
        if (values.shape != (3,) or np.any(~np.isfinite(values))
                or values[0] <= 0.0 or values[1] <= 0.0 or values[2] == 0.0):
            raise ValueError(
                f"species[{index}] density/mass must be positive and charge non-zero, all finite"
            )
        density.append(values[0])
        mass.append(values[1])
        charge.append(values[2])

    k = float(wave_number_m_inverse)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("wave_number_m_inverse must be positive and finite")
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty vector")
    if np.any(omega == 0.0):
        raise ValueError(
            "zero frequency requires a separately converged static-limit extrapolation"
        )

    if isinstance(quadrature_order, (bool, np.bool_)) or int(quadrature_order) != quadrature_order or quadrature_order < 24:
        raise ValueError("quadrature_order must be an integer >= 24")
    if isinstance(quadrature_max_order, (bool, np.bool_)) or int(quadrature_max_order) != quadrature_max_order or quadrature_max_order <= quadrature_order:
        raise ValueError("quadrature_max_order must be an integer > quadrature_order")
    tolerance = float(quadrature_convergence_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("quadrature_convergence_tolerance must be positive and finite")

    current_order = int(quadrature_order)
    bare = conserving_bgk_longitudinal_susceptibility(
        k, omega,
        species_densities_m3=density,
        species_masses_kg=mass,
        common_temperature_J=common_temperature_J,
        collision_frequencies_s_inverse=collision_frequencies_s_inverse,
        bulk_flow_m_s=bulk_flow_m_s,
        quadrature_order=current_order,
    )
    response0 = np.asarray(bare["density_response_matrix_m3_per_J"], dtype=complex)
    quadrature_error = np.inf
    quadrature_converged = False
    quadrature_history = [current_order]
    while current_order < int(quadrature_max_order):
        next_order = min(2 * current_order, int(quadrature_max_order))
        refined = conserving_bgk_longitudinal_susceptibility(
            k, omega,
            species_densities_m3=density,
            species_masses_kg=mass,
            common_temperature_J=common_temperature_J,
            collision_frequencies_s_inverse=collision_frequencies_s_inverse,
            bulk_flow_m_s=bulk_flow_m_s,
            quadrature_order=next_order,
        )
        refined_response = np.asarray(
            refined["density_response_matrix_m3_per_J"], dtype=complex
        )
        scale = max(float(np.linalg.norm(refined_response)), np.finfo(float).tiny)
        quadrature_error = float(np.linalg.norm(refined_response - response0) / scale)
        quadrature_history.append(next_order)
        bare = refined
        response0 = refined_response
        current_order = next_order
        if quadrature_error <= tolerance:
            quadrature_converged = True
            break
    interaction = np.outer(charge, charge) / (epsilon_0 * k**2)
    identity = np.eye(len(species), dtype=complex)
    screened = np.empty_like(response0)
    dielectric = np.empty_like(response0)
    condition = np.empty(omega.size)
    for index, response in enumerate(response0):
        dielectric[index] = identity - response @ interaction
        condition[index] = np.linalg.cond(dielectric[index])
        if not np.isfinite(condition[index]) or condition[index] > 1e14:
            raise ValueError(
                "BGK CTS dielectric matrix is singular or ill-conditioned; "
                "inspect frequency and increase velocity quadrature resolution"
            )
        screened[index] = np.linalg.solve(dielectric[index], response)

    fdt = classical_fdt_dynamic_structure_factor(
        omega, screened, temperature_J=common_temperature_J,
        psd_relative_tolerance=psd_relative_tolerance,
    )
    return {
        **bare,
        "bare_density_response_m3_per_J": response0,
        "screened_density_response_m3_per_J": screened,
        "dielectric_matrix": dielectric,
        "dielectric_condition_number": condition,
        "velocity_quadrature_convergence": {
            "orders_evaluated": quadrature_history,
            "final_order": current_order,
            "relative_response_change": quadrature_error,
            "relative_tolerance": tolerance,
            "converged": quadrature_converged,
        },
        "coulomb_interaction_matrix_J_m3": interaction,
        "dynamic_structure_factor_m3_s": fdt["dynamic_structure_factor_m3_s"],
        "fdt_closure": fdt["closure"],
        "fdt_positive_semidefinite": fdt["positive_semidefinite"],
        "screening_model": "self_consistent_multispecies_electrostatic_Poisson",
        "cts_model": "isothermal_momentum_conserving_multispecies_BGK_classical_FDT",
        "integration_status": "BGK_response_Poisson_screening_and_matching_classical_FDT_connected",
        "quantitative_release_gate": {
            "velocity_quadrature_convergence_required": True,
            "velocity_quadrature_converged": quadrature_converged,
            "collision_model_validation_required": True,
            "frequency_resolution_convergence_required": True,
            "velocity_quadrature_release_allowed": quadrature_converged,
            "quantitative_spectrum_release_allowed": False,
            "quantitative_release_blockers": [
                "Independent collision-model validation is still required.",
                "Frequency-grid convergence is still required.",
                "Experimental scattering/receiver normalization is not included.",
            ],
            "experimental_validation_performed": False,
        },
        "limitations": list(bare["limitations"]) + list(fdt["limitations"]) + [
            "Electrostatic longitudinal response only; not magnetized Full-EM CTS.",
            "BGK is an isothermal closure, not the velocity-dependent Coulomb Landau operator.",
            "Classical equilibrium FDT only; quantum detailed balance is not included.",
            "Absolute receiver power requires calibrated scattering geometry and instrument response.",
        ],
    }
