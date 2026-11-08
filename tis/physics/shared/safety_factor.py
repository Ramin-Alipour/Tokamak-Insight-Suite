"""Safety-factor estimates from supplied equilibrium geometry and fields."""
from __future__ import annotations

import numpy as np


def estimate_safety_factor(
    R_m, minor_radius_m, B_toroidal_T, B_poloidal_T, *,
    q_profile=None, q_profile_rho=None, q_profile_psi_n=None
):
    """Estimate cylindrical q and summarize an optional q profile.

    q_profile_rho is normalized minor radius and yields q(rho=0.95), not q95.
    q95 is conventionally evaluated at normalized poloidal flux psi_N=0.95;
    use q_profile_psi_n for that result. Coordinates must be supplied explicitly.
    """
    R = np.asarray(R_m, dtype=float)
    a = np.asarray(minor_radius_m, dtype=float)
    bt = np.asarray(B_toroidal_T, dtype=float)
    bp = np.asarray(B_poloidal_T, dtype=float)
    if (
        any(not np.all(np.isfinite(x)) for x in (R, a, bt, bp))
        or np.any(R <= 0)
        or np.any(a <= 0)
        or np.any(bp <= 0)
    ):
        raise ValueError("Geometry and field values must be finite; R, a and Bp positive")
    R, a, bt, bp = np.broadcast_arrays(R, a, bt, bp)
    if np.any(a >= R):
        raise ValueError("Minor radius must be smaller than major radius")

    q = a * bt / (R * bp)
    out = {
        "q_cylindrical_estimate": q,
        "equation": "q(r) approximately r*B_t/(R*B_p) for circular, large-aspect-ratio geometry",
        "caveat": (
            "Approximate circular large-aspect-ratio estimate only; accurate q(r) "
            "requires equilibrium flux-surface integration with consistent local "
            "major radius and poloidal field."
        ),
    }

    if q_profile_rho is not None and q_profile_psi_n is not None:
        raise ValueError("Provide only one profile coordinate: q_profile_rho or q_profile_psi_n")
    coordinate = q_profile_psi_n if q_profile_psi_n is not None else q_profile_rho
    coordinate_name = "q_profile_psi_n" if q_profile_psi_n is not None else "q_profile_rho"
    if coordinate is not None and q_profile is None:
        raise ValueError(f"{coordinate_name} requires q_profile")
    if q_profile is not None:
        qp = np.asarray(q_profile, dtype=float)
        if qp.ndim != 1 or len(qp) < 2 or not np.all(np.isfinite(qp)):
            raise ValueError("q_profile must contain at least two finite 1-D samples")
        out["provided_q_profile"] = qp
        out["q_min_provided"] = float(np.min(qp))
        if coordinate is None:
            out["q95_note"] = (
                "q95 not computed: supply q_profile_psi_n (normalized poloidal flux); "
                "profile sample indices are not a physical coordinate."
            )
        else:
            coord = np.asarray(coordinate, dtype=float)
            if (
                coord.ndim != 1
                or coord.shape != qp.shape
                or not np.all(np.isfinite(coord))
                or np.any(np.diff(coord) <= 0)
                or coord[0] < 0
                or coord[-1] > 1
            ):
                raise ValueError(
                    f"{coordinate_name} must match q_profile, be finite and strictly "
                    "increasing, and lie within [0, 1]"
                )
            if coord[0] > 0.95 or coord[-1] < 0.95:
                raise ValueError(f"{coordinate_name} must cover coordinate=0.95")
            value = float(np.interp(0.95, coord, qp))
            out[coordinate_name] = coord
            if q_profile_psi_n is not None:
                out["q95_provided"] = value
                out["q95_coordinate"] = "normalized poloidal flux psi_N=0.95"
                out["q95_interpolation_method"] = "linear interpolation at supplied psi_N=0.95 coordinate"
            else:
                out["q_at_rho_0_95"] = value
                out["profile_coordinate_note"] = (
                    "This is q at normalized minor radius rho=0.95, not necessarily conventional q95 "
                    "(which is generally defined at normalized poloidal flux psi_N=0.95)."
                )
    return out


def safety_factor_from_field_line_pitch(theta_rad, dphi_dtheta):
    """Compute q=(1/2pi) integral(dphi/dtheta dtheta) over one poloidal turn."""
    theta = np.asarray(theta_rad, dtype=float)
    slope = np.asarray(dphi_dtheta, dtype=float)
    if theta.ndim != 1 or slope.shape != theta.shape or len(theta) < 3:
        raise ValueError("theta_rad and dphi_dtheta must be equal-length 1-D arrays with at least 3 points")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(slope)) or np.any(np.diff(theta) <= 0):
        raise ValueError("Inputs must be finite and theta strictly increasing")
    span = float(theta[-1] - theta[0])
    if not np.isclose(span, 2 * np.pi, rtol=1e-3, atol=1e-6):
        raise ValueError("theta_rad must cover one full 2*pi poloidal turn")
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    q = float(integrate(slope, theta) / (2 * np.pi))
    return {
        "safety_factor_q": q,
        "toroidal_turns_per_poloidal_turn": q,
        "equation": "q=(1/(2*pi))*integral_0^(2*pi) (dphi/dtheta) dtheta",
    }
