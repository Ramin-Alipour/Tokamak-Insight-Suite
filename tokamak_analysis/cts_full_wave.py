"""One-dimensional, full-vector Maxwell solver for stratified magnetized plasma.

Solves the tangential-field Berreman system without a paraxial or scalar-wave
approximation. The dielectric tensor is supplied at every z sample, allowing
complex, anisotropic and mode-coupling media. This is a 1-D slab solver: it
does not represent transverse diffraction or arbitrary 3-D interfaces.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C = 1.0 / np.sqrt(EPS0 * MU0)


def _dielectric_stack(z_m, epsilon_r):
    z = np.asarray(z_m, dtype=float)
    eps = np.asarray(epsilon_r, dtype=complex)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("z_m must be a finite, strictly increasing 1-D grid")
    if eps.shape != (z.size, 3, 3) or np.any(~np.isfinite(eps)):
        raise ValueError("epsilon_r must have shape (len(z_m), 3, 3) and be finite")
    if np.any(np.abs(eps[:, 2, 2]) < 1e-12):
        raise ValueError("epsilon_zz is singular; this tangential-field formulation cannot eliminate Ez")
    return z, eps


def berreman_matrix(omega_rad_s, epsilon_r):
    """Return A for d[Ex,Ey,Hx,Hy]/dz = A [Ex,Ey,Hx,Hy]."""
    w = float(omega_rad_s)
    e = np.asarray(epsilon_r, dtype=complex)
    if not np.isfinite(w) or w <= 0 or e.shape != (3, 3) or np.any(~np.isfinite(e)):
        raise ValueError("omega_rad_s must be positive and epsilon_r a finite 3x3 tensor")
    if abs(e[2, 2]) < 1e-12:
        raise ValueError("epsilon_zz is singular")
    # D_z=0 for a plane-stratified medium; eliminate Ez exactly.
    ez_ex, ez_ey = -e[2, 0] / e[2, 2], -e[2, 1] / e[2, 2]
    dxx = e[0, 0] + e[0, 2] * ez_ex
    dxy = e[0, 1] + e[0, 2] * ez_ey
    dyx = e[1, 0] + e[1, 2] * ez_ex
    dyy = e[1, 1] + e[1, 2] * ez_ey
    # exp(-i omega t); H is in A/m.
    return np.array([
        [0, 0, 0, 1j*w*MU0],
        [0, 0, -1j*w*MU0, 0],
        [-1j*w*EPS0*dyx, -1j*w*EPS0*dyy, 0, 0],
        [1j*w*EPS0*dxx, 1j*w*EPS0*dxy, 0, 0],
    ], dtype=complex)


def solve_stratified_full_wave(omega_rad_s, z_m, epsilon_r, *,
                               incident_amplitude_V_m=(1.0, 0.0),
                               rtol=1e-8, atol=1e-10, max_step_m=None):
    """Solve a normally incident wave on a finite anisotropic plasma slab.

    Vacuum is assumed on both sides. Incident Ex/Ey amplitudes are specified
    at z[0]. The solver obtains reflected and transmitted tangential fields by
    matching to forward/backward vacuum plane waves at both boundaries.
    Complex epsilon includes absorption. Returned mode conversion is coherent
    and is therefore not approximated by independent scalar power channels.
    """
    z, eps = _dielectric_stack(z_m, epsilon_r)
    w = float(omega_rad_s)
    if not np.isfinite(w) or w <= 0:
        raise ValueError("omega_rad_s must be positive and finite")
    inc = np.asarray(incident_amplitude_V_m, dtype=complex)
    if inc.shape != (2,) or np.any(~np.isfinite(inc)):
        raise ValueError("incident_amplitude_V_m must be a finite complex 2-vector")
    k0 = w / C
    # State transfer matrix, integrated as a 4x4 fundamental matrix.
    def rhs(s, flat):
        e_local = np.array([[np.interp(s, z, eps[:, i, j].real)
                             + 1j*np.interp(s, z, eps[:, i, j].imag)
                             for j in range(3)] for i in range(3)])
        return (berreman_matrix(w, e_local) @ flat.reshape(4, 4)).ravel()
    solver_options = {"method": "DOP853", "rtol": rtol, "atol": atol}
    if max_step_m is not None:
        if not np.isfinite(max_step_m) or max_step_m <= 0:
            raise ValueError("max_step_m must be positive and finite when supplied")
        solver_options["max_step"] = float(max_step_m)
    sol = solve_ivp(rhs, (z[0], z[-1]), np.eye(4, dtype=complex).ravel(),
                    **solver_options)
    if not sol.success:
        raise RuntimeError("full-wave fundamental-matrix integration failed: " + sol.message)
    transfer = sol.y[:, -1].reshape(4, 4)
    # Vacuum state for each tangential E polarization, forward and backward.
    q = 1.0 / (MU0 * C)
    forward = np.array([[1,0],[0,1],[0,-q],[q,0]], dtype=complex)
    backward = np.array([[1,0],[0,1],[0,q],[-q,0]], dtype=complex)
    left_rhs = np.vstack((np.eye(2, dtype=complex), np.zeros((2,2), dtype=complex)))
    # At z0: state = forward*a + backward*r. At zL: state=forward*t.
    lhs = np.column_stack((-transfer @ backward, forward))
    rhs_vec = transfer @ forward @ inc
    solution = np.linalg.solve(lhs, rhs_vec)
    reflected, transmitted = solution[:2], solution[2:]
    state0 = forward @ inc + backward @ reflected
    stateL = forward @ transmitted
    # In vacuum, normal time-averaged Poynting flux is |E|^2/(2 Z0).
    pin = float(np.vdot(inc, inc).real / (2*MU0*C))
    pref = float(np.vdot(reflected, reflected).real / (2*MU0*C))
    ptrans = float(np.vdot(transmitted, transmitted).real / (2*MU0*C))
    return {
        "z_m": z.copy(), "fundamental_matrix": transfer,
        "incident_electric_amplitude_V_m": inc,
        "reflected_electric_amplitude_V_m": reflected,
        "transmitted_electric_amplitude_V_m": transmitted,
        "incident_power_flux_W_m2": pin,
        "reflected_power_flux_W_m2": pref,
        "transmitted_power_flux_W_m2": ptrans,
        "reflectance": pref/pin, "transmittance": ptrans/pin,
        "absorptance": 1.0-(pref+ptrans)/pin,
        "boundary_state_left": state0, "boundary_state_right": stateL,
        "solver": "DOP853_integrated_full_vector_Berreman_4x4",
        "approximation_scope": "1-D stratified slab, normal incidence, local supplied dielectric tensor; no transverse diffraction or 3-D interfaces",
        "mode_conversion": "coherent vector coupling retained in the full 4x4 Maxwell system",
    }
