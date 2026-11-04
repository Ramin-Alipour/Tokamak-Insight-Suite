"""Classical fluctuation-dissipation closure for collisional density response.

Converts a validated retarded density susceptibility into its classical
equilibrium spectral covariance. It does not construct a Landau collision
operator. A BGK response passed by the caller remains BGK, not Landau.
"""
from __future__ import annotations
import numpy as np


def density_fluctuation_spectral_matrix(
    density_response_matrix_m3_per_J, angular_frequency_rad_s, temperature_J,
    *, zero_frequency_imaginary_slope=None, positivity_tolerance=1e-9,
):
    r"""Classical matrix FDT for delta_n = chi @ delta_U, exp(-i omega t).

    S_nn(omega) = -(2 T / omega) Im_H[chi(omega)],
    Im_H[chi] = (chi - chi^H)/(2 i).

    At zero frequency the limit is -2 T d Im_H[chi]/d omega and its slope
    must be supplied explicitly. No low-frequency floor is invented.
    """
    raw = np.asarray(density_response_matrix_m3_per_J)
    if raw.dtype.kind not in "fc":
        raise ValueError("density response must be a finite complex-valued array")
    chi = np.asarray(raw, dtype=complex)
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    temperature = float(temperature_J)
    tolerance = float(positivity_tolerance)
    if chi.ndim != 3 or chi.shape[1] != chi.shape[2] or chi.shape[0] == 0:
        raise ValueError("density response must have shape (n_frequency, n_species, n_species)")
    if omega.ndim != 1 or omega.shape[0] != chi.shape[0] or np.any(~np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be finite and match the response frequency axis")
    if np.any(~np.isfinite(chi)):
        raise ValueError("density response must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_J must be positive and finite")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("positivity_tolerance must be finite and non-negative")

    imag_h = (chi - np.swapaxes(chi.conj(), 1, 2)) / (2j)
    imag_h = 0.5 * (imag_h + np.swapaxes(imag_h.conj(), 1, 2))
    spectrum = np.empty_like(imag_h)
    zero = omega == 0.0
    if np.any(zero):
        if zero_frequency_imaginary_slope is None:
            raise ValueError("zero-frequency FDT limit requires zero_frequency_imaginary_slope")
        slope = np.asarray(zero_frequency_imaginary_slope, dtype=complex)
        count = int(np.count_nonzero(zero))
        if slope.shape == chi.shape[1:]:
            slope = np.broadcast_to(slope, (count,) + slope.shape)
        elif slope.shape != (count,) + chi.shape[1:]:
            raise ValueError("zero-frequency slope must be (n_species,n_species) or (n_zero,n_species,n_species)")
        if np.any(~np.isfinite(slope)):
            raise ValueError("zero-frequency slope must be finite")
        slope_h = 0.5 * (slope + np.swapaxes(slope.conj(), 1, 2))
        spectrum[zero] = -2.0 * temperature * slope_h

    nonzero = ~zero
    spectrum[nonzero] = -2.0 * temperature * imag_h[nonzero] / omega[nonzero, None, None]
    minimum_eigenvalues = np.empty(omega.size, dtype=float)
    for index, matrix in enumerate(spectrum):
        hermitian = 0.5 * (matrix + matrix.conj().T)
        eigenvalues = np.linalg.eigvalsh(hermitian)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        minimum_eigenvalues[index] = float(eigenvalues[0])
        if eigenvalues[0] < -tolerance * scale:
            raise ValueError(
                "FDT spectral matrix is not positive semidefinite at frequency "
                f"index {index}; minimum eigenvalue={eigenvalues[0]:.6e}"
            )
        if eigenvalues[0] < 0.0:
            values, vectors = np.linalg.eigh(hermitian)
            spectrum[index] = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T

    return {
        "density_fluctuation_spectral_matrix_m3_s": spectrum,
        "angular_frequency_rad_s": omega.copy(),
        "temperature_J": temperature,
        "fdt": "classical_two_sided_matrix_fluctuation_dissipation",
        "response_convention": "delta_n=chi@delta_U; exp(-i*omega*t)",
        "zero_frequency_limit": "explicit_imaginary_susceptibility_slope" if np.any(zero) else "not_needed",
        "minimum_eigenvalue_before_roundoff_projection": minimum_eigenvalues,
        "positive_semidefinite": True,
        "collision_operator_status": "FDT_closure_only_operator_not_constructed",
        "limitations": [
            "Requires a retarded response from a separately specified kinetic operator.",
            "Does not implement the velocity-dependent linearized Landau test-particle and field-particle terms.",
            "Does not construct the velocity-space Langevin source covariance for a Landau solver.",
            "Classical limit only; quantum detailed balance is not included.",
            "A BGK response passed by the caller remains BGK and is not a Landau result.",
        ],
    }
