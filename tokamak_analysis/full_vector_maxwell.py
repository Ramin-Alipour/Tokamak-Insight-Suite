"""Full-vector non-paraxial Maxwell propagation in a stratified anisotropic plasma.

The solver evolves all four tangential field components.  It retains backward
waves, reflection and polarization coupling; unlike a paraxial beam/ray model,
it makes no small-angle expansion.  The medium is assumed invariant in x,y
and is supplied as a local 3x3 relative dielectric tensor epsilon(z).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as C_LIGHT, mu_0 as MU0, epsilon_0 as EPS0


def propagate_full_vector_stratified_maxwell(
    z_m,
    transverse_wavevector_rad_m,
    angular_frequency_rad_s,
    relative_permeability,
    relative_dielectric_tensor,
    initial_tangential_fields,
    *,
    samples=201,
    rtol=1e-8,
    atol=1e-10,
) -> dict:
    r"""Integrate the exact 1-D stratified Maxwell system for arbitrary kx,ky.

    Fields vary as exp(i kx x + i ky y - i omega t).  The state is
    (Ex,Ey,Hx,Hy); Ez and Hz are eliminated algebraically from Maxwell's
    z-components.  A complex tensor callback epsilon(z) permits gyrotropy
    and off-diagonal mode coupling.  The solver includes forward/backward
    branches and is therefore not a one-way coupled-mode approximation.
    """
    z = np.asarray(z_m, dtype=float)
    kt = np.asarray(transverse_wavevector_rad_m, dtype=float)
    state0 = np.asarray(initial_tangential_fields, dtype=complex)
    omega, mur = float(angular_frequency_rad_s), float(relative_permeability)
    if z.shape != (2,) or not np.all(np.isfinite(z)) or z[1] <= z[0]:
        raise ValueError("z_m must contain finite increasing [z_start,z_end]")
    if kt.shape != (2,) or not np.all(np.isfinite(kt)):
        raise ValueError("transverse_wavevector_rad_m must be a finite [kx,ky] vector")
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(mur) or mur <= 0:
        raise ValueError("frequency and relative permeability must be positive and finite")
    if state0.shape != (4,) or not np.all(np.isfinite(state0.real)) or not np.all(np.isfinite(state0.imag)):
        raise ValueError("initial_tangential_fields must be a finite complex [Ex,Ey,Hx,Hy] vector")
    if isinstance(samples, bool) or int(samples) != samples or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol) or atol <= 0:
        raise ValueError("rtol and atol must be positive and finite")
    mu = MU0 * mur
    kx, ky = map(float, kt)

    def matrix_at(position):
        eps = np.asarray(relative_dielectric_tensor(float(position)), dtype=complex)
        if eps.shape != (3, 3) or not np.all(np.isfinite(eps.real)) or not np.all(np.isfinite(eps.imag)):
            raise ValueError("relative_dielectric_tensor(z) must return a finite complex 3x3 tensor")
        if abs(eps[2, 2]) <= 1e-13 * max(float(np.linalg.norm(eps)), 1.0):
            raise ValueError("epsilon_zz is singular; this reduced tangential formulation cannot cross the longitudinal resonance")
        return eps

    def rhs(position, state):
        ex, ey, hx, hy = state
        eps = matrix_at(position)
        hz = (kx * ey - ky * ex) / (omega * mu)
        dz = (ky * hx - kx * hy) / omega
        ez = (dz - eps[2, 0] * ex - eps[2, 1] * ey) / eps[2, 2]
        dx = EPS0 * (eps[0, 0] * ex + eps[0, 1] * ey + eps[0, 2] * ez)
        dy = EPS0 * (eps[1, 0] * ex + eps[1, 1] * ey + eps[1, 2] * ez)
        return np.array([
            1j * kx * ez + 1j * omega * mu * hy,
            1j * ky * ez - 1j * omega * mu * hx,
            1j * kx * hz - 1j * omega * dy,
            1j * ky * hz + 1j * omega * dx,
        ], dtype=complex)

    grid = np.linspace(z[0], z[1], int(samples))
    solution = solve_ivp(rhs, (z[0], z[1]), state0, t_eval=grid,
                         method="DOP853", rtol=float(rtol), atol=float(atol))
    if not solution.success:
        raise RuntimeError(f"full-vector Maxwell propagation failed: {solution.message}")
    tangential = solution.y.T
    full = np.empty((grid.size, 6), dtype=complex)
    for i, (position, state) in enumerate(zip(grid, tangential)):
        ex, ey, hx, hy = state
        eps = matrix_at(position)
        hz = (kx * ey - ky * ex) / (omega * mu)
        dz = (ky * hx - kx * hy) / omega
        ez = (dz - eps[2, 0] * ex - eps[2, 1] * ey) / eps[2, 2]
        full[i] = [ex, ey, ez, hx, hy, hz]
    poynting_z = 0.5 * np.real(
        full[:, 0] * full[:, 4].conj() - full[:, 1] * full[:, 3].conj()
    )
    return {
        "z_m": grid,
        "electric_field_V_m": full[:, :3],
        "magnetic_field_A_m": full[:, 3:],
        "tangential_state": tangential,
        "poynting_flux_z_W_m2": poynting_z,
        "solver": "full_vector_stratified_Maxwell_DAE_reduced_to_four_tangential_components",
        "paraxial_approximation": False,
        "backward_waves_and_reflection_retained": True,
        "polarization_coupling_retained": True,
        "mode_conversion": "resolved_by_full_vector_field_evolution_when_epsilon(z)_couples_modes",
        "assumptions": [
            "Medium is invariant in x and y; only z dependence is represented.",
            "Local linear constitutive law D=epsilon_0 epsilon(z) E and scalar mu_r.",
            "The supplied dielectric tensor and initial fields define the physical problem.",
            "No nonlinear response, transverse beam envelope or finite-aperture diffraction is included.",
        ],
        "limitations": [
            "Cannot cross epsilon_zz=0 with this algebraic elimination; use a formulation regular at the longitudinal resonance.",
            "No boundary matching is performed automatically; provide physically consistent initial tangential fields.",
            "Mode conversion at unresolved resonant/cutoff layers requires mesh and solver convergence studies.",
            "No equilibrium, dielectric tensor or launch conditions are inferred.",
        ],
        "integration_success": True,
        "function_evaluations": int(solution.nfev),
    }
