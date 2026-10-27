"""Axisymmetric poloidal magnetic-flux reconstruction from magnetic-field grids.

Uses psi(R,Z)-psi(R0,Z0) = integral(R B_Z dR - R B_R dZ).
Input fields must already be calibrated in tesla on a rectangular R-Z grid.
The returned flux is relative to a reference point unless an initial flux is supplied.
"""
from __future__ import annotations
import numpy as np


def reconstruct_poloidal_flux(R_m, Z_m, B_R_T, B_Z_T, *, initial_flux_Wb=0.0,
                              reference_index=None):
    R = np.asarray(R_m, dtype=float)
    Z = np.asarray(Z_m, dtype=float)
    br = np.asarray(B_R_T, dtype=float)
    bz = np.asarray(B_Z_T, dtype=float)
    if R.ndim != 1 or Z.ndim != 1 or R.size < 2 or Z.size < 2:
        raise ValueError("R_m and Z_m must be 1-D coordinate arrays with at least two points each.")
    if not all(np.all(np.isfinite(x)) for x in (R, Z, br, bz)):
        raise ValueError("Coordinates and magnetic-field grids must contain only finite values.")
    if np.any(R <= 0):
        raise ValueError("R_m must be strictly positive for axisymmetric poloidal-flux reconstruction.")
    if np.any(np.diff(R) <= 0) or np.any(np.diff(Z) <= 0):
        raise ValueError("R_m and Z_m must be strictly increasing and unique.")
    expected = (Z.size, R.size)
    if br.shape != expected or bz.shape != expected:
        raise ValueError(f"B_R_T and B_Z_T must both have shape (len(Z_m), len(R_m)) = {expected}.")
    if isinstance(initial_flux_Wb, (bool, np.bool_)) or not np.isscalar(initial_flux_Wb) or not np.isfinite(initial_flux_Wb):
        raise ValueError("initial_flux_Wb must be a finite scalar.")
    if reference_index is None:
        iz0, ir0 = 0, 0
    else:
        if (not isinstance(reference_index, (tuple, list)) or len(reference_index) != 2
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in reference_index)):
            raise ValueError("reference_index must be (z_index, r_index) integers.")
        iz0, ir0 = map(int, reference_index)
        if not (0 <= iz0 < Z.size and 0 <= ir0 < R.size):
            raise ValueError("reference_index is outside the field grid.")
    # Integrate each row in R, then each column in Z. The reference value fixes
    # the additive gauge; this path is deterministic and makes no equilibrium claim.
    dpsi_dR = R[None, :] * bz
    psi_rows = np.zeros_like(bz, dtype=float)
    psi_rows[:, 1:] = np.cumsum(0.5 * (dpsi_dR[:, 1:] + dpsi_dR[:, :-1]) * np.diff(R)[None, :], axis=1)
    dpsi_dZ = -R[None, :] * br
    psi = psi_rows.copy()
    for j in range(1, Z.size):
        psi[j, :] = psi[j-1, :] + 0.5 * (dpsi_dZ[j, :] + dpsi_dZ[j-1, :]) * (Z[j]-Z[j-1])
    psi = psi - psi[iz0, ir0] + float(initial_flux_Wb)
    # Local consistency residual compares the two independent derivative constraints.
    dpsi_dR_num = np.gradient(psi, R, axis=1, edge_order=1)
    dpsi_dZ_num = np.gradient(psi, Z, axis=0, edge_order=1)
    residual_R = dpsi_dR_num - R[None, :] * bz
    residual_Z = dpsi_dZ_num + R[None, :] * br
    return {
        "R_m": R.copy(), "Z_m": Z.copy(), "poloidal_flux_Wb": psi,
        "poloidal_flux_change_Wb": psi - float(initial_flux_Wb),
        "initial_flux_Wb": float(initial_flux_Wb),
        "reference_index_z_r": [iz0, ir0],
        "dpsi_dR_Wb_m": dpsi_dR_num, "dpsi_dZ_Wb_m": dpsi_dZ_num,
        "consistency_residual_R_Wb_m": residual_R,
        "consistency_residual_Z_Wb_m": residual_Z,
        "diagnostics": {
            "grid_shape": list(psi.shape), "sample_count": int(psi.size),
            "flux_min_Wb": float(np.min(psi)), "flux_max_Wb": float(np.max(psi)),
            "flux_peak_to_peak_Wb": float(np.ptp(psi)),
            "rms_derivative_residual_R_Wb_m": float(np.sqrt(np.mean(residual_R**2))),
            "rms_derivative_residual_Z_Wb_m": float(np.sqrt(np.mean(residual_Z**2))),
            "absolute_flux_depends_on_initial_condition": True,
            "input_fields_assumed_calibrated_T": True,
        },
        "assumptions": [
            "Axisymmetric cylindrical coordinates (R,Z); B_R and B_Z are calibrated in tesla.",
            "Flux is reconstructed by path integration on the supplied rectangular grid.",
            "The reference point sets the additive flux gauge; absolute flux needs an independent boundary/reference condition.",
            "This is not an EFIT/Grad-Shafranov equilibrium solver and does not infer plasma current or pressure profiles.",
            "Field inconsistency and measurement noise can make the path-integrated map path-dependent; inspect derivative residuals.",
        ],
    }


analyze_poloidal_flux = reconstruct_poloidal_flux
