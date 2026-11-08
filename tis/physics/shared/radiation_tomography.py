"""Regularized linear reconstruction of emissivity from line-integrated signals."""
import numpy as np


def reconstruct_emissivity(response_matrix, line_integrals, *, regularization=1e-6, nonnegative=True):
    A = np.asarray(response_matrix, dtype=float)
    y = np.asarray(line_integrals, dtype=float)
    lam = float(regularization)
    if A.ndim != 2 or y.ndim != 1 or A.shape[0] != len(y) or min(A.shape) < 1 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)):
        raise ValueError("Finite response matrix and matching measurements required")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("Regularization must be nonnegative")
    augA = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]) if lam > 0 else A
    augy = np.r_[y, np.zeros(A.shape[1])] if lam > 0 else y
    if nonnegative:
        from scipy.optimize import lsq_linear
        sol = lsq_linear(augA, augy, bounds=(0, np.inf)).x
    else:
        sol = np.linalg.lstsq(augA, augy, rcond=None)[0]
    residual = A @ sol - y
    return {
        "emissivity": sol,
        "synthetic_measurements": A @ sol,
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "regularization": lam,
        "caveat": "Reconstruction quality depends on calibrated geometry matrix, coverage, regularization, and uncertainty model.",
    }


def abel_project_axisymmetric_emissivity(radius_m, emissivity_W_m3, impact_parameter_m):
    """Forward Abel projection using u=sqrt(r^2-p^2) to remove the endpoint singularity."""
    r = np.asarray(radius_m, dtype=float)
    eps = np.asarray(emissivity_W_m3, dtype=float)
    p = np.asarray(impact_parameter_m, dtype=float)
    if r.ndim != 1 or eps.shape != r.shape or len(r) < 3 or p.ndim != 1 or len(p) < 1 or any(not np.all(np.isfinite(v)) for v in (r, eps, p)):
        raise ValueError("Radius/emissivity must match and all inputs be finite 1-D arrays")
    if np.any(np.diff(r) <= 0) or not np.isclose(r[0], 0.0, atol=1e-12) or np.any(eps < 0) or np.any(p < 0) or np.any(p >= r[-1]):
        raise ValueError("Require radius from zero, non-negative emissivity and impact parameters in [0,a)")
    values = []
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    for chord in p:
        # The substitution u=sqrt(r^2-p^2) turns
        # epsilon(r)*r/sqrt(r^2-p^2) dr into epsilon(sqrt(u^2+p^2)) du,
        # including the formerly singular lower endpoint u=0.
        radial_path = np.concatenate(([chord], r[r > chord]))
        u = np.sqrt(np.maximum(radial_path**2 - chord**2, 0.0))
        emissivity_path = np.interp(radial_path, r, eps)
        values.append(float(2 * integrate(emissivity_path, u)))
    return {
        "impact_parameter_m": p,
        "line_integrated_emissivity_W_m2": np.asarray(values),
        "equation": "g(p)=2*integral_p^a epsilon(r)*r/sqrt(r^2-p^2) dr",
        "caveat": "Discrete forward Abel projection using a singularity-removing coordinate transform and linear interpolation; resolution and emissivity interpolation still affect accuracy.",
    }


def forward_project_emissivity(response_matrix, emissivity):
    """Forward tomography model y=A*epsilon, mapping voxel emissivity to line-integrated measurements."""
    A = np.asarray(response_matrix, dtype=float)
    eps = np.asarray(emissivity, dtype=float)
    if A.ndim != 2 or eps.ndim != 1 or A.shape[1] != len(eps) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(eps)):
        raise ValueError("Finite 2-D response matrix and matching 1-D emissivity vector required")
    if np.any(eps < 0):
        raise ValueError("Emissivity must be non-negative")
    return {
        "predicted_line_integrals": A @ eps,
        "equation": "y=A*epsilon",
        "caveat": "Response matrix must encode calibrated line-of-sight geometry and detector response.",
    }


def reconstruct_bolometric_tomography(bolometer_response_matrix, bolometer_signals, *, regularization=1e-6, nonnegative=True):
    """Bolometer-specific emissivity reconstruction using a calibrated bolometer response matrix."""
    result = reconstruct_emissivity(
        bolometer_response_matrix, bolometer_signals,
        regularization=regularization, nonnegative=nonnegative,
    )
    result["diagnostic_type"] = "bolometric tomography"
    result["caveat"] = "Requires a calibrated bolometer view/response matrix and compatible signal units; generic inversion does not supply equilibrium mapping or detector calibration."
    return result
