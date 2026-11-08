"""Multi-pin probe channel synchronization, potential gradients and E-field estimate."""
from __future__ import annotations

import numpy as np


def analyze_multipin_probe(
    time_s,
    potentials_V,
    positions_m,
    *,
    axis="poloidal",
    potential_covariance_V2=None,
):
    """Estimate E=-dV/dx and propagate correlated voltage measurement noise.

    potential_covariance_V2 may be one shared (n_pin,n_pin) covariance matrix
    or one (n_time,n_pin,n_pin) matrix per time sample. Covariances are
    absolute and include correlations between pins. They are propagated
    through the exact finite-difference operator used for the reported
    gradient: C_E = D C_V D.T. Cross-time covariance and position uncertainty
    are not represented by this contract.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(potentials_V, dtype=float)
    x = np.asarray(positions_m, dtype=float)
    if (
        t.ndim != 1 or v.ndim != 2 or v.shape[0] != len(t)
        or v.shape[1] != len(x) or len(x) < 2
    ):
        raise ValueError("Potentials must have shape (time, pins) and match positions")
    if (
        not np.all(np.isfinite(t)) or not np.all(np.isfinite(v))
        or not np.all(np.isfinite(x)) or np.any(np.diff(t) <= 0)
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("Finite data and strictly increasing time/position required")

    edge_order = 2 if len(x) > 2 else 1
    derivative_operator = np.gradient(
        np.eye(len(x), dtype=float), x, axis=0, edge_order=edge_order
    )
    gradient = v @ derivative_operator.T
    result = {
        "time_s": t,
        "positions_m": x,
        "potential_V": v,
        "potential_gradient_V_m": gradient,
        "electric_field_V_m": -gradient,
        "axis": axis,
        "derivative_operator_per_m": derivative_operator,
        "mean_potential_by_pin_V": np.mean(v, axis=0),
        "equation": "E_axis = -D_x V; C_E = D_x C_V D_x.T",
        "caveat": (
            "Electric-field estimate assumes simultaneous calibrated potentials "
            "and known pin coordinates; sheath effects and spatial resolution remain."
        ),
    }

    if potential_covariance_V2 is not None:
        covariance = np.asarray(potential_covariance_V2, dtype=float)
        n_time, n_pin = v.shape
        if covariance.shape == (n_pin, n_pin):
            covariance = np.broadcast_to(covariance, (n_time, n_pin, n_pin))
            covariance_scope = "shared_pin_covariance"
        elif covariance.shape == (n_time, n_pin, n_pin):
            covariance_scope = "time_resolved_pin_covariance"
        else:
            raise ValueError(
                "potential_covariance_V2 must have shape (n_pin,n_pin) "
                "or (n_time,n_pin,n_pin)"
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("potential_covariance_V2 must contain finite values")
        covariance_scale = max(
            float(np.max(np.abs(covariance))), np.finfo(float).tiny
        )
        asymmetry = float(np.max(np.abs(covariance - covariance.transpose(0, 2, 1))))
        if asymmetry > 1.0e-10 * covariance_scale:
            raise ValueError("potential_covariance_V2 must be symmetric")
        covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigen_scale = np.maximum(
            np.max(np.abs(eigenvalues), axis=1), np.finfo(float).tiny
        )
        if np.any(eigenvalues[:, 0] < -1.0e-10 * eigen_scale):
            raise ValueError("potential_covariance_V2 must be positive semidefinite")

        field_covariance = np.einsum(
            "ij,tjk,lk->til", derivative_operator, covariance,
            derivative_operator,
        )
        field_variance = np.diagonal(field_covariance, axis1=1, axis2=2)
        variance_scale = np.maximum(
            np.max(np.abs(field_variance), axis=1, keepdims=True),
            np.finfo(float).tiny,
        )
        if np.any(field_variance < -1.0e-10 * variance_scale):
            raise ArithmeticError("propagated electric-field variance is materially negative")
        result.update({
            "potential_covariance_V2": covariance,
            "electric_field_covariance_V2_m2": field_covariance,
            "electric_field_standard_uncertainty_V_m": np.sqrt(
                np.maximum(field_variance, 0.0)
            ),
            "uncertainty_scope": covariance_scope,
            "uncertainty_method": "linear_covariance_propagation_through_reported_finite_difference_operator",
        })
    return result


def exb_drift_velocity(E_x_V_m, E_y_V_m, E_z_V_m, B_x_T, B_y_T, B_z_T):
    """Compute ideal E-cross-B drift v=E×B/B^2, perpendicular to B."""
    ex, ey, ez, bx, by, bz = np.broadcast_arrays(*[
        np.asarray(v, dtype=float)
        for v in (E_x_V_m, E_y_V_m, E_z_V_m, B_x_T, B_y_T, B_z_T)
    ])
    if any(not np.all(np.isfinite(v)) for v in (ex, ey, ez, bx, by, bz)):
        raise ValueError("Field components must be finite")
    b2 = bx*bx + by*by + bz*bz
    if np.any(b2 <= 0):
        raise ValueError("Magnetic field magnitude must be non-zero")
    vx = (ey*bz - ez*by) / b2
    vy = (ez*bx - ex*bz) / b2
    vz = (ex*by - ey*bx) / b2
    return {
        "v_exb_x_m_s": vx,
        "v_exb_y_m_s": vy,
        "v_exb_z_m_s": vz,
        "equation": "v_E=E×B/|B|^2",
        "caveat": "Ideal guiding-center E-cross-B drift; requires colocated calibrated E and B components.",
    }
