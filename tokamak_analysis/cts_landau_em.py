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
    direction = direction / np.linalg.norm(direction)
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


def landau_full_em_electric_field_fluctuation_spectrum(
    constitutive_response: Mapping,
    wave_number_m_inverse: float,
    *,
    propagation_direction=(1.0, 0.0, 0.0),
    relative_dielectric_tensor=None,
    condition_limit: float = 1.0e12,
) -> dict:
    """Propagate Landau--FDT current noise through the local Maxwell Green dyadic.

    For exp(-i omega t), D = k^2 I - k k^T - (omega/c)^2 epsilon_r,
    G = D^-1, E = i mu0 omega G J and S_E = mu0^2 omega^2 G S_J G^dagger.
    Faraday's law then gives B = (k x E)/omega and its full spectral tensor.
    This is a local homogeneous-medium fluctuation kernel, not a magnetized
    CTS three-wave vertex, ray solution, or calibrated receiver spectrum.
    """
    if not isinstance(constitutive_response, Mapping):
        raise TypeError("constitutive_response must be a mapping from the Landau-FDT Full-EM solver")
    omega = np.asarray(constitutive_response.get("angular_frequency_rad_s"), dtype=float)
    current_noise = np.array(constitutive_response.get("current_fluctuation_spectral_matrix"), dtype=complex, copy=True)
    epsilon = np.asarray(
        relative_dielectric_tensor if relative_dielectric_tensor is not None
        else constitutive_response.get("relative_dielectric_tensor"), dtype=complex
    )
    k = float(wave_number_m_inverse)
    direction = np.asarray(propagation_direction, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)) or np.any(omega <= 0):
        raise ValueError("angular_frequency_rad_s must be a positive finite vector")
    if current_noise.shape != (omega.size, 3, 3) or np.any(~np.isfinite(current_noise)):
        raise ValueError("current_fluctuation_spectral_matrix must have shape (n_frequency,3,3)")
    if epsilon.shape == (3, 3):
        epsilon = np.broadcast_to(epsilon, (omega.size, 3, 3))
    if epsilon.shape != (omega.size, 3, 3) or np.any(~np.isfinite(epsilon)):
        raise ValueError("relative_dielectric_tensor must have shape (3,3) or (n_frequency,3,3)")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("wave_number_m_inverse must be positive and finite")
    if direction.shape != (3,) or np.any(~np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("propagation_direction must be a finite non-zero 3-vector")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    direction = direction / np.linalg.norm(direction)
    # For exp(-i omega t) and omega>0, a passive background must have
    # Im_H(epsilon_r)=(epsilon_r-epsilon_r^dagger)/(2i) positive semidefinite.
    # Reject gain media instead of returning a finite but physically inconsistent
    # equilibrium fluctuation spectrum.
    for index, tensor in enumerate(epsilon):
        dissipative = (tensor - tensor.conj().T) / (2j)
        dissipative = 0.5 * (dissipative + dissipative.conj().T)
        loss_eigenvalues = np.linalg.eigvalsh(dissipative)
        loss_scale = max(float(np.linalg.norm(dissipative, ord=2)), 1.0)
        if loss_eigenvalues[0] < -1e-10 * loss_scale:
            raise ValueError(
                f"relative_dielectric_tensor[{index}] is active: "
                "Im_H(epsilon_r) must be positive semidefinite for exp(-i*omega*t)"
            )
    for index, matrix in enumerate(current_noise):
        scale = max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
        if np.linalg.norm(matrix - matrix.conj().T, ord=2) > 1e-10 * scale:
            raise ValueError(f"current_fluctuation_spectral_matrix[{index}] must be Hermitian")
        hermitian = 0.5 * (matrix + matrix.conj().T)
        if np.linalg.eigvalsh(hermitian)[0] < -1e-10 * scale:
            raise ValueError(f"current_fluctuation_spectral_matrix[{index}] must be positive semidefinite")
        current_noise[index] = hermitian
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    analyzer_1 = np.cross(direction, reference)
    analyzer_1 /= np.linalg.norm(analyzer_1)
    analyzer_2 = np.cross(direction, analyzer_1)
    analyzer_basis = np.column_stack((analyzer_1, analyzer_2))
    kvec = k * direction
    mu0, c0 = 4.0e-7*np.pi, 299792458.0
    eye = np.eye(3, dtype=complex)
    cross_k = np.array([
        [0.0, -kvec[2], kvec[1]],
        [kvec[2], 0.0, -kvec[0]],
        [-kvec[1], kvec[0], 0.0],
    ], dtype=complex)
    levi_civita = np.zeros((3, 3, 3), dtype=float)
    levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1.0
    levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1.0
    green = np.empty_like(current_noise)
    electric_noise = np.empty_like(current_noise)
    magnetic_noise = np.empty_like(current_noise)
    electric_magnetic_cross = np.empty_like(current_noise)
    poynting = np.empty((omega.size, 3), dtype=float)
    conditions = np.empty(omega.size)
    for i, w in enumerate(omega):
        operator = k*k*eye - np.outer(kvec, kvec) - (w/c0)**2*epsilon[i]
        conditions[i] = np.linalg.cond(operator)
        if not np.isfinite(conditions[i]) or conditions[i] > condition_limit:
            raise np.linalg.LinAlgError(
                f"Maxwell Green operator is singular/ill-conditioned at frequency index {i}; "
                "include physical damping or move away from the undamped mode"
            )
        green[i] = np.linalg.solve(operator, eye)
        electric_noise[i] = (mu0*w)**2 * green[i] @ current_noise[i] @ green[i].conj().T
        electric_noise[i] = 0.5*(electric_noise[i] + electric_noise[i].conj().T)
        magnetic_map = cross_k / w
        magnetic_noise[i] = magnetic_map @ electric_noise[i] @ magnetic_map.conj().T
        magnetic_noise[i] = 0.5*(magnetic_noise[i] + magnetic_noise[i].conj().T)
        electric_magnetic_cross[i] = electric_noise[i] @ magnetic_map.conj().T
        poynting[i] = 0.5/mu0*np.real(
            np.einsum("ijk,jk->i", levi_civita, electric_magnetic_cross[i])
        )
    for name, tensor in (("electric", electric_noise), ("magnetic", magnetic_noise)):
        eigenvalues = np.linalg.eigvalsh(tensor)
        scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), np.finfo(float).tiny)
        if np.any(eigenvalues[:, 0] < -1e-9*scale):
            raise ArithmeticError(f"{name} field fluctuation matrix is not positive semidefinite")
    coherency = np.einsum("ai,nab,bj->nij", analyzer_basis.conj(), electric_noise, analyzer_basis)
    coherency = 0.5 * (coherency + np.swapaxes(coherency.conj(), 1, 2))
    c11, c22 = np.real(coherency[:, 0, 0]), np.real(coherency[:, 1, 1])
    c12 = coherency[:, 0, 1]
    stokes = np.column_stack((c11+c22, c11-c22, 2*np.real(c12), 2*np.imag(c12)))
    intensity = stokes[:, 0]
    degree = np.divide(np.linalg.norm(stokes[:, 1:], axis=1), intensity,
        out=np.zeros_like(intensity), where=intensity > 0.0)
    if np.any(degree > 1.0 + 1e-8):
        raise ArithmeticError("transverse Stokes vector violates the coherency positivity bound")
    return {
        "angular_frequency_rad_s": omega.copy(),
        "wave_number_m_inverse": k,
        "propagation_direction": direction,
        "maxwell_green_dyadic_m2": green,
        "electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s": electric_noise,
        "magnetic_field_fluctuation_spectral_matrix_T2_per_rad_s": magnetic_noise,
        "electric_magnetic_cross_spectral_matrix": electric_magnetic_cross,
        "spectral_poynting_flux_W_m2_per_rad_s": poynting,
        "transverse_analyzer_basis_cartesian": analyzer_basis,
        "transverse_electric_coherency_spectral_matrix_V2_m2_per_rad_s": coherency,
        "transverse_stokes_spectral_density_IQUV": stokes,
        "degree_of_linear_and_circular_polarization": degree,
        "stokes_convention": "I=C11+C22; Q=C11-C22; U=2 Re(C12); V=2 Im(C12); exp(-i*omega*t)",
        "maxwell_operator_condition_number": conditions,
        "field_fluctuation_relation": "S_E = mu_0^2 omega^2 G S_J G^dagger",
        "magnetic_fluctuation_relation": "S_B = (k_cross/omega) S_E (k_cross/omega)^dagger; B = k x E / omega",
        "spectral_poynting_relation": "Pi = 0.5 Re <E x B*>_omega",
        "time_convention": "exp(-i*omega*t)",
        "integration_status": "Landau-FDT current noise propagated through local Maxwell Green dyadic and Faraday closure",
        "scope": "local homogeneous linear dielectric; frequency-diagonal classical field fluctuations",
        "cts_three_wave_vertex_connected": False,
        "limitations": [
            "No magnetized kinetic CTS three-wave vertex or ray/beam propagation.",
            "No cross-frequency noise covariance is inferred.",
            "Undamped Maxwell poles are rejected by the condition-number gate.",
            "Absolute receiver power requires geometry, etendue, calibration and background transfer.",
        ],
    }
