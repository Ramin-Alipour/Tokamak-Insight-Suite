"""Finite-difference Grad–Shafranov solver with prescribed source terms and boundary flux."""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_grad_shafranov(R_m, Z_m, source_Wb_per_m2, boundary_psi_Wb) -> dict:
    """Solve Δ*ψ = source on an R-Z grid with fixed Dirichlet boundary ψ.

    Δ*ψ = d²ψ/dR² - (1/R)dψ/dR + d²ψ/dZ². The source term must already
    represent -μ0 R² p'(ψ) - F F'(ψ) in a consistent convention. This is a
    forward equilibrium solve for prescribed source profiles, not an EFIT-like
    inverse reconstruction from magnetic measurements.
    """
    R = np.asarray(R_m, dtype=float).ravel()
    Z = np.asarray(Z_m, dtype=float).ravel()
    source = np.asarray(source_Wb_per_m2, dtype=float)
    boundary = np.asarray(boundary_psi_Wb, dtype=float)
    if R.size < 3 or Z.size < 3 or np.any(~np.isfinite(R)) or np.any(~np.isfinite(Z)):
        raise ValueError("R_m and Z_m must be finite grids with at least three points each.")
    if np.any(np.diff(R) <= 0) or np.any(np.diff(Z) <= 0) or np.any(R <= 0):
        raise ValueError("R_m and Z_m must be strictly increasing, and R_m positive.")
    drs, dzs = np.diff(R), np.diff(Z)
    if not np.allclose(drs, drs[0], rtol=1e-5, atol=1e-12) or not np.allclose(dzs, dzs[0], rtol=1e-5, atol=1e-12):
        raise ValueError("This finite-difference solver currently requires uniform R-Z spacing.")
    expected = (Z.size, R.size)
    if source.shape != expected or boundary.shape != expected:
        raise ValueError(f"source_Wb_per_m2 and boundary_psi_Wb must both have shape {expected}.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(boundary)):
        raise ValueError("Source and boundary flux arrays must be finite.")

    nr, nz = R.size, Z.size
    dri, dzi = float(drs[0]), float(dzs[0])
    n_unknown = (nr - 2) * (nz - 2)
    matrix = lil_matrix((n_unknown, n_unknown), dtype=float)
    rhs = np.zeros(n_unknown, dtype=float)

    def idx(j, i):
        return (j - 1) * (nr - 2) + (i - 1)

    for j in range(1, nz - 1):
        for i in range(1, nr - 1):
            row = idx(j, i)
            r = R[i]
            c_r2 = 1.0 / dri**2
            c_r1 = -1.0 / (2.0 * r * dri)
            c_z2 = 1.0 / dzi**2
            center = -2.0 * c_r2 - 2.0 * c_z2
            coeffs = [
                (j, i, center),
                (j, i + 1, c_r2 + c_r1),
                (j, i - 1, c_r2 - c_r1),
                (j + 1, i, c_z2),
                (j - 1, i, c_z2),
            ]
            rhs[row] = source[j, i]
            for jj, ii, coeff in coeffs:
                if 1 <= jj < nz - 1 and 1 <= ii < nr - 1:
                    matrix[row, idx(jj, ii)] += coeff
                else:
                    rhs[row] -= coeff * boundary[jj, ii]
    solution = spsolve(matrix.tocsr(), rhs)
    psi = boundary.copy()
    psi[1:-1, 1:-1] = solution.reshape((nz - 2, nr - 2))
    residual = np.zeros_like(psi)
    residual[1:-1, 1:-1] = (
        (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dri**2
        - (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * R[None, 1:-1] * dri)
        + (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dzi**2
        - source[1:-1, 1:-1]
    )
    return {
        "diagnostic": "Grad–Shafranov forward equilibrium solve",
        "R_m": R,
        "Z_m": Z,
        "poloidal_flux_Wb": psi,
        "source_term_Wb_per_m2": source,
        "interior_residual_rms": float(np.sqrt(np.mean(residual[1:-1, 1:-1] ** 2))),
        "interior_residual_max_abs": float(np.max(np.abs(residual[1:-1, 1:-1]))),
        "solver": "sparse finite-difference Dirichlet solve",
        "assumptions_and_limits": [
            "Forward solution for supplied source terms and boundary flux; not an inverse EFIT reconstruction.",
            "Uniform rectangular R-Z grid and axisymmetric Grad–Shafranov operator assumed.",
            "Physical equilibrium quality depends on source profiles, boundary conditions and calibration supplied by the caller.",
        ],
    }
