"""Polarized magnetoionic radiative transfer for CTS backgrounds.

Implements the constant-coefficient Stokes transfer equation
    dS/ds = j - M S
with S=(I,Q,U,V), including dichroic absorption and Faraday rotation/
conversion.  The Mueller propagation matrix follows Leung, Gammie & Noble,
ApJ 737, 21 (2011), Eqs. (7)-(8), DOI:10.1088/0004-637X/737/1/21.

The coefficients are physical inputs from a magnetoionic dielectric/emission
solver. This module does not invent them from |B| or replace that solver.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (4,):
        raise ValueError(f"{name} must be a finite real length-4 vector")
    result = raw.astype(float, copy=False)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def stokes_mueller_absorption_matrix(
    alpha_I_m_inverse, alpha_Q_m_inverse, alpha_U_m_inverse,
    alpha_V_m_inverse, rho_Q_m_inverse, rho_U_m_inverse,
    rho_V_m_inverse,
):
    """Build the Stokes/Mueller propagation matrix M [m^-1].

    rho_Q, rho_U, rho_V are the linear/circular birefringence coefficients
    (Faraday conversion/rotation) in the convention of Leung et al. (2011).
    """
    values = np.asarray([
        alpha_I_m_inverse, alpha_Q_m_inverse, alpha_U_m_inverse,
        alpha_V_m_inverse, rho_Q_m_inverse, rho_U_m_inverse,
        rho_V_m_inverse,
    ])
    if values.dtype.kind not in "iuf" or np.any(~np.isfinite(values)):
        raise ValueError("all Mueller coefficients must be finite real scalars")
    ai, aq, au, av, rq, ru, rv = values.astype(float)
    dichroism = np.sqrt(aq*aq + au*au + av*av)
    if ai < 0.0 or dichroism > ai + 1e-12*max(ai, dichroism, np.finfo(float).tiny):
        raise ValueError(
            "passive-medium absorption requires alpha_I >= "
            "sqrt(alpha_Q^2 + alpha_U^2 + alpha_V^2) >= 0"
        )
    return np.array([
        [ai, aq, au, av],
        [aq, ai, rv, -ru],
        [au, -rv, ai, rq],
        [av, ru, -rq, ai],
    ], dtype=float)


def propagate_stokes_slab(
    incident_stokes, emissivity_stokes, propagation_matrix_m_inverse,
    path_length_m,
):
    """Propagate Stokes radiance through a homogeneous slab exactly.

    Solves the affine constant-coefficient system with one augmented matrix
    exponential; this remains well-defined when M is singular or has zero
    eigenvalues, unlike a formula that explicitly inverts M.

    Parameters
    ----------
    incident_stokes : array-like, shape (4,)
        Incident (I,Q,U,V) spectral radiance.
    emissivity_stokes : array-like, shape (4,)
        (j_I,j_Q,j_U,j_V), in radiance per metre.
    propagation_matrix_m_inverse : array-like, shape (4,4)
        Mueller absorption/birefringence matrix in inverse metres.
    path_length_m : float
        Positive homogeneous path length in metres.
    """
    incident = _vector(incident_stokes, "incident_stokes")
    emissivity = _vector(emissivity_stokes, "emissivity_stokes")
    matrix = np.asarray(propagation_matrix_m_inverse)
    if matrix.dtype.kind not in "iuf" or matrix.shape != (4, 4):
        raise ValueError("propagation_matrix_m_inverse must be a finite real 4x4 matrix")
    matrix = matrix.astype(float, copy=False)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("propagation_matrix_m_inverse must be finite")
    length_raw = np.asarray(path_length_m)
    if length_raw.dtype.kind not in "iuf" or length_raw.ndim != 0:
        raise ValueError("path_length_m must be a finite positive real scalar")
    length = float(length_raw)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("path_length_m must be finite and positive")
    if incident[0] < 0.0 or incident[0] + 1e-12 < np.linalg.norm(incident[1:]):
        raise ValueError("incident Stokes vector is not physically realizable: I >= sqrt(Q^2+U^2+V^2) >= 0")
    if emissivity[0] < 0.0 or emissivity[0] + 1e-12 < np.linalg.norm(emissivity[1:]):
        raise ValueError("emissivity Stokes vector is not physically realizable")
    augmented = np.zeros((5, 5), dtype=float)
    augmented[:4, :4] = -matrix
    augmented[:4, 4] = emissivity
    state = np.concatenate((incident, [1.0]))
    output = expm(augmented * length) @ state
    stokes = output[:4]
    # Numerical roundoff may produce a tiny negative polarization-cone margin.
    cone_margin = stokes[0] - np.linalg.norm(stokes[1:])
    if stokes[0] < -1e-12 or cone_margin < -1e-9*max(abs(stokes[0]), 1.0):
        raise ArithmeticError("propagated Stokes vector left the physical polarization cone")
    return {
        "stokes_radiance": stokes,
        "I": float(stokes[0]),
        "Q": float(stokes[1]),
        "U": float(stokes[2]),
        "V": float(stokes[3]),
        "path_length_m": length,
        "transfer_model": "exact_constant_coefficient_Stokes_Mueller_matrix_exponential",
        "included_effects": [
            "polarization-dependent absorption (dichroism)",
            "Faraday rotation",
            "Faraday conversion",
            "polarized emission",
        ],
        "coefficient_provenance": "externally supplied magnetoionic coefficients; not inferred from scalar magnetic-field magnitude",
    }
