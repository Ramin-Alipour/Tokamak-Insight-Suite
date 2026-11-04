"""Electromagnetic constitutive response from the conservative Landau--FDT operator.

Projects the multispecies Landau resolvent and its matching Langevin source
onto Cartesian current components. This is a local unmagnetized, zero-flow,
common-temperature Maxwellian Full-EM constitutive kernel, not a CTS
three-wave scattering vertex. Magnetized, relativistic and non-Maxwellian
equilibria are not approximated.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from .cts_landau_fdt import LandauFDTSystem, landau_density_response

EPS0 = 8.8541878128e-12


def landau_full_em_current_response(
    system: LandauFDTSystem,
    angular_frequency_rad_s,
    *,
    species: Sequence[Mapping],
    common_temperature_J: float,
    wave_number_m_inverse: float,
    propagation_direction=(1.0, 0.0, 0.0),
) -> dict:
    """Return collisional conductivity, dielectric and FDT current-noise tensors."""
    if not isinstance(system, LandauFDTSystem):
        raise TypeError("system must be a LandauFDTSystem")
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)) or not species:
        raise ValueError("species must be a non-empty sequence")
    if len(species) != len(system.species_slices):
        raise ValueError("species count must match the assembled Landau system")
    temperature = float(common_temperature_J)
    k = float(wave_number_m_inverse)
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    direction = np.asarray(propagation_direction, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("common_temperature_J must be positive and finite")
    if not np.isclose(temperature, system.metadata["common_temperature_J"], rtol=1e-12, atol=0.0):
        raise ValueError("common_temperature_J must match the assembled Landau equilibrium")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("wave_number_m_inverse must be positive and finite")
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)) or np.any(omega == 0):
        raise ValueError("angular_frequency_rad_s must be a finite non-zero vector")
    if direction.shape != (3,) or np.any(~np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("propagation_direction must be a finite non-zero 3-vector")
    direction /= np.linalg.norm(direction)
    flow = np.asarray(system.metadata["common_bulk_flow_m_s"], dtype=float)
    if flow.shape != (3,) or np.any(~np.isfinite(flow)) or np.linalg.norm(flow) > 1e-10:
        raise ValueError("Full-EM Landau bridge currently requires a zero-flow equilibrium")
    weights = np.asarray(system.equilibrium_weights, dtype=float)
    velocity = np.asarray(system.velocity_grid_m_s, dtype=float)
    nv = system.species_slices[0].stop - system.species_slices[0].start
    if velocity.shape != (len(species), nv, 3) or weights.shape != (len(species)*nv,):
        raise ValueError("Landau system velocity-grid/weight shape is inconsistent")
    current_observation = np.zeros((3, weights.size), dtype=float)
    electric_drive = np.zeros((weights.size, 3), dtype=complex)
    reference_species = system.metadata["species"]
    for index, (item, sl, ref) in enumerate(zip(species, system.species_slices, reference_species)):
        if not isinstance(item, Mapping):
            raise ValueError(f"species[{index}] must be a mapping")
        actual = np.asarray([item.get("mass_kg", np.nan), item.get("charge_C", np.nan), item.get("density_m3", np.nan)], dtype=float)
        expected = np.asarray([ref["mass_kg"], ref["charge_C"], ref["density_m3"]], dtype=float)
        if np.any(~np.isfinite(actual)) or actual[0] <= 0 or actual[1] == 0 or actual[2] <= 0:
            raise ValueError(f"species[{index}] mass, charge and density must be physical")
        if not np.allclose(actual, expected, rtol=1e-12, atol=0.0):
            raise ValueError(f"species[{index}] must match the species used to assemble the Landau system")
        v = velocity[index]
        w = weights[sl]
        charge = actual[1]
        current_observation[:, sl] = charge * w[None, :] * v.T
        electric_drive[sl, :] = (charge / temperature) * v
    projected = landau_density_response(
        system, omega, current_observation, forcing=electric_drive,
        wave_number_m_inverse=k, propagation_direction=direction,
    )
    conductivity = np.asarray(projected["response"], dtype=complex)
    noise = np.asarray(projected["fluctuation_spectral_matrix"], dtype=complex)
    noise = 0.5 * (noise + np.swapaxes(noise.conj(), 1, 2))
    if conductivity.shape != (omega.size, 3, 3) or noise.shape != (omega.size, 3, 3):
        raise ArithmeticError("Landau current projection returned an unexpected tensor shape")
    if np.any(~np.isfinite(conductivity)) or np.any(~np.isfinite(noise)):
        raise ArithmeticError("Landau current response contains non-finite values")
    eigenvalues = np.linalg.eigvalsh(noise)
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), np.finfo(float).tiny)
    if np.any(np.min(eigenvalues, axis=1) < -1e-9*scale):
        raise ArithmeticError("projected Landau-FDT current-noise matrix is not positive semidefinite")
    dielectric = np.eye(3, dtype=complex)[None, :, :] + 1j*conductivity/(EPS0*omega[:, None, None])
    return {
        "angular_frequency_rad_s": omega.copy(),
        "wave_number_m_inverse": k,
        "propagation_direction": direction,
        "conductivity_tensor_S_m": conductivity,
        "relative_dielectric_tensor": dielectric,
        "current_fluctuation_spectral_matrix": noise,
        "current_noise_eigenvalues": eigenvalues,
        "collision_operator": system.metadata["operator"],
        "fdt_source": system.metadata["fdt"],
        "response_solver": "three_cartesian_drives_sparse_Landau_resolvent",
        "dielectric_convention": "exp(-i*omega*t); epsilon_r=I+i*sigma/(epsilon_0*omega)",
        "full_em_scope": "unmagnetized_zero_flow_common_temperature_Maxwellian_constitutive_response",
        "cts_three_wave_vertex_connected": False,
        "quantitative_release_gate": {
            "velocity_grid_convergence_required": True,
            "velocity_domain_convergence_required": True,
            "frequency_resolution_convergence_required": True,
            "experimental_validation_performed": False,
        },
        "limitations": [
            "Constitutive Full-EM response/noise only; not a magnetized CTS three-wave scattering vertex.",
            "Unmagnetized, zero-flow, common-temperature isotropic Maxwellian equilibrium only.",
            "Non-relativistic classical Landau operator; no quantum detailed balance.",
            "Finite velocity-domain and grid convergence are mandatory.",
            "No ray tracing, mode conversion or absolute scattered-power normalization.",
        ],
    }
