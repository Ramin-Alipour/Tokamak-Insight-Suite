"""Classical fluctuation--dissipation closure for CTS response matrices.

This module implements the *closure*, not a collision operator.  In particular,
it must only be applied to a causal susceptibility computed with the same
collision model and equilibrium.  It does not turn BGK or an ad-hoc damping
width into a Landau calculation.
"""
from __future__ import annotations

import numpy as np


def classical_fdt_dynamic_structure_factor(
    angular_frequency_rad_s,
    retarded_density_response_m3_per_J,
    *,
    temperature_J: float,
    psd_relative_tolerance: float = 1e-8,
) -> dict:
    r"""Apply the classical matrix FDT to a retarded density susceptibility.

    For the exp(-i omega t) convention and a response to an external potential
    energy, the symmetrized classical spectrum is
        S(omega) = -(2 k_B T / omega) Im_H[chi(omega)],
    where Im_H chi = (chi-chi^H)/(2i).  The input is expected to have shape
    (n_omega,n_species,n_species), with SI units m^-3/J.  The result has units
    m^-3 s.  Zero frequency is deliberately rejected: a static limit must be
    obtained by a separately converged low-frequency extrapolation, not by
    division by zero or an invented value.
    """
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    chi = np.asarray(retarded_density_response_m3_per_J, dtype=complex)
    temperature = float(temperature_J)
    tolerance = float(psd_relative_tolerance)
    if omega.ndim != 1 or omega.size == 0 or not np.all(np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty vector")
    if np.any(omega == 0.0):
        raise ValueError("zero frequency requires a separately converged static-limit extrapolation")
    if chi.ndim != 3 or chi.shape[0] != omega.size or chi.shape[1] != chi.shape[2] or chi.shape[1] == 0:
        raise ValueError("retarded response must have shape (n_frequency,n_species,n_species)")
    if not np.all(np.isfinite(chi.real)) or not np.all(np.isfinite(chi.imag)):
        raise ValueError("retarded response must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_J must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("psd_relative_tolerance must be positive and finite")
    antihermitian_imaginary = (chi - np.swapaxes(chi.conj(), 1, 2)) / (2j)
    spectrum = -(2.0 * temperature / omega[:, None, None]) * antihermitian_imaginary
    spectrum = 0.5 * (spectrum + np.swapaxes(spectrum.conj(), 1, 2))
    minimum_eigenvalues = np.empty(omega.size)
    for index, matrix in enumerate(spectrum):
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        minimum_eigenvalues[index] = float(eigenvalues[0])
        if eigenvalues[0] < -tolerance * scale:
            raise ValueError(
                "FDT spectrum is not positive semidefinite; check response causality, "
                "Fourier/sign convention, collision closure and numerical convergence"
            )
        if eigenvalues[0] < 0.0:
            values, vectors = np.linalg.eigh(matrix)
            spectrum[index] = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T
    return {
        "dynamic_structure_factor_m3_s": spectrum,
        "angular_frequency_rad_s": omega.copy(),
        "temperature_J": temperature,
        "closure": "classical_matrix_fluctuation_dissipation_theorem",
        "response_fourier_convention": "exp(-i*omega*t)",
        "response_units": "m^-3/J",
        "spectrum_units": "m^-3*s",
        "minimum_eigenvalue_before_roundoff_projection": minimum_eigenvalues,
        "positive_semidefinite": True,
        "collision_operator_status": "inherited_from_input_susceptibility_not_computed_here",
        "limitations": [
            "This is a fluctuation-dissipation closure only; it does not compute a susceptibility or a Landau collision operator.",
            "The supplied susceptibility must be causal and use the same equilibrium, species, collisions and Fourier convention.",
            "Classical limit only; quantum detailed-balance corrections are not included.",
            "The zero-frequency limit must be obtained by a separately converged extrapolation.",
        ],
    }



def quantum_fdt_dynamic_structure_factor(
    angular_frequency_rad_s,
    retarded_density_response_m3_per_J,
    *,
    temperature_J: float,
    psd_relative_tolerance: float = 1e-8,
) -> dict:
    r"""Apply the symmetrized quantum matrix FDT to a retarded susceptibility.

    For exp(-i omega t), with response to an external potential energy,
        S_sym(omega) = -hbar*coth(hbar*omega/(2*T))*Im_H[chi(omega)].
    Here temperature_J denotes k_B*T in joules. The classical limit is
    -(2*T/omega)*Im_H[chi]. This is the symmetrized quantum spectrum; it is
    not the unsymmetrized neutron-scattering structure factor S^>(omega).
    """
    from scipy.constants import hbar

    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    chi = np.asarray(retarded_density_response_m3_per_J, dtype=complex)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty vector")
    if np.any(omega == 0.0):
        raise ValueError(
            "zero frequency requires a separately converged static-limit extrapolation"
        )
    if (chi.ndim != 3 or chi.shape[0] != omega.size
            or chi.shape[1] != chi.shape[2] or chi.shape[1] == 0
            or np.any(~np.isfinite(chi.real)) or np.any(~np.isfinite(chi.imag))):
        raise ValueError(
            "retarded response must be finite with shape (n_frequency,n_species,n_species)"
        )
    if isinstance(temperature_J, (bool, np.bool_)):
        raise ValueError("temperature_J must be positive and finite")
    temperature = float(temperature_J)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_J must be positive and finite")
    if isinstance(psd_relative_tolerance, (bool, np.bool_)):
        raise ValueError("psd_relative_tolerance must be positive and finite")
    tolerance = float(psd_relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("psd_relative_tolerance must be positive and finite")

    y = hbar * omega / (2.0 * temperature)
    abs_y = np.abs(y)
    quantum_factor = np.empty_like(y)
    small = abs_y < 1.0e-4
    ys = y[small]
    # Evaluate hbar*coth(y) as (2*T/omega)*(y*coth(y)); this
    # remains finite even if the dimensionless y underflows to exactly zero.
    quantum_factor[small] = (2.0 * temperature / omega[small]) * (
        1.0 + ys**2 / 3.0 - ys**4 / 45.0
    )
    large = ~small
    a = abs_y[large]
    coth_y = np.sign(y[large]) * (
        1.0 + 2.0 * np.exp(-2.0 * a) / (-np.expm1(-2.0 * a))
    )
    quantum_factor[large] = hbar * coth_y

    imaginary = (chi - np.swapaxes(chi.conj(), 1, 2)) / (2j)
    spectrum = -quantum_factor[:, None, None] * imaginary
    spectrum = 0.5 * (spectrum + np.swapaxes(spectrum.conj(), 1, 2))
    minimum_eigenvalues = np.empty(omega.size)
    for index, matrix in enumerate(spectrum):
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        minimum_eigenvalues[index] = float(eigenvalues[0])
        if eigenvalues[0] < -tolerance * scale:
            raise ValueError(
                "quantum FDT spectrum is not positive semidefinite; check response "
                "causality, Fourier/sign convention and numerical convergence"
            )
        if eigenvalues[0] < 0.0:
            values, vectors = np.linalg.eigh(matrix)
            spectrum[index] = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T

    return {
        "dynamic_structure_factor_m3_s": spectrum,
        "angular_frequency_rad_s": omega.copy(),
        "temperature_J": temperature,
        "closure": "symmetrized_quantum_matrix_fluctuation_dissipation_theorem",
        "response_fourier_convention": "exp(-i*omega*t)",
        "response_units": "m^-3/J",
        "spectrum_units": "m^-3*s",
        "quantum_argument_hbar_omega_over_2T": y,
        "minimum_eigenvalue_before_roundoff_projection": minimum_eigenvalues,
        "positive_semidefinite": True,
        "detailed_balance_representation": "symmetrized_spectrum_only",
        "limitations": [
            "Returns the symmetrized quantum spectrum, not the unsymmetrized S-greater or S-less spectrum.",
            "The supplied retarded susceptibility must be causal and use the same equilibrium, species, collisions and Fourier convention.",
            "A classical susceptibility is not upgraded to a quantum dielectric by this FDT prefactor.",
            "Zero frequency requires a separately converged static-limit extrapolation.",
        ],
    }
