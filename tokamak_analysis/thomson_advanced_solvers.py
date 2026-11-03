"""Advanced kinetic ECE and full-vector stratified Maxwell solvers.

The ECE conductivity uses the existing magnetized-Maxwellian Vlasov kernel.
Its damping input is a numerical resonance regularizer, not a conserving
Landau collision operator. The Maxwell solver is exact for a medium varying
only along z; it includes polarization coupling but not transverse diffraction.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from .cts_full_em import _species_conductivity
from .cts_em_geometry import solve_cold_plasma_propagating_modes

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


def kinetic_ece_absorption_coefficient(
    frequency_Hz: float, propagation_direction, magnetic_field_T,
    species: Sequence[dict], *, mode_index: int,
    harmonic_order: int = 12, quadrature_order: int = 32,
    damping_rate_rad_s: float,
) -> dict:
    """Mode-resolved Maxwellian kinetic absorption alpha in inverse metres.

    The cold-plasma eigenvector is normalized to unit longitudinal Poynting
    flux. alpha = Re(E* sigma E) is then dissipated power density per flux.
    The explicit damping is numerical resonance regularization only.
    """
    f = float(frequency_Hz)
    if not np.isfinite(f) or f <= 0:
        raise ValueError("frequency_Hz must be positive and finite")
    if isinstance(mode_index, bool) or mode_index not in (0, 1):
        raise ValueError("mode_index must explicitly select branch 0 or 1")
    if isinstance(harmonic_order, bool) or int(harmonic_order) < 1:
        raise ValueError("harmonic_order must be a positive integer")
    if isinstance(quadrature_order, bool) or int(quadrature_order) < 8:
        raise ValueError("quadrature_order must be an integer >= 8")
    damping = float(damping_rate_rad_s)
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping_rate_rad_s must be positive and finite")
    direction = np.asarray(propagation_direction, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("propagation_direction must be a finite non-zero 3-vector")
    direction /= np.linalg.norm(direction)
    omega = 2 * np.pi * f
    mode = solve_cold_plasma_propagating_modes(
        omega, direction, magnetic_field_T, species, mode_index=int(mode_index))
    k = np.asarray(mode["wavevector_rad_m"], dtype=float)
    electric = np.asarray(mode["electric_polarization_unit_flux"], dtype=complex)
    sigma = np.zeros((3, 3), dtype=complex)
    species_power = {}
    for i, item in enumerate(species):
        required = {"mass", "charge", "density", "temperature", "flow"}
        if not required.issubset(item):
            raise ValueError(f"species[{i}] requires {sorted(required)}")
        contribution = _species_conductivity(
            omega, k, magnetic_field_T, item,
            harmonic_order=int(harmonic_order),
            quadrature_order=int(quadrature_order),
            damping_rate_rad_s=damping)
        sigma += contribution
        species_power[str(item.get("name", i))] = float(
            np.real(np.vdot(electric, contribution @ electric)))
    absorbed = float(np.real(np.vdot(electric, sigma @ electric)))
    scale = max(float(np.linalg.norm(sigma) * np.vdot(electric, electric).real), 1.0)
    if absorbed < -1e-9 * scale:
        raise ValueError("negative absorption: non-passive/gain response detected")
    return {
        "frequency_Hz": f, "mode_index": int(mode_index),
        "refractive_index": mode["refractive_index"],
        "wavevector_rad_m": k, "electric_polarization_unit_flux": electric,
        "conductivity_S_m": sigma,
        "absorption_coefficient_m_inverse": max(absorbed, 0.0),
        "species_absorbed_power_density_per_flux_m_inverse": species_power,
        "kinetic_model": "magnetized_Maxwellian_Vlasov_conductivity",
        "regularization": {"damping_rate_rad_s": damping,
                           "interpretation": "numerical resonance regularizer; not Landau collisions"},
        "limitations": [
            "Cold local eigenmode with kinetic Maxwellian dissipative conductivity.",
            "No arbitrary relativistic/non-Maxwellian EVDF.",
            "No conserving collision operator or FDT fluctuation source.",
            "No inhomogeneous path mode conversion in this absorption-only function.",
        ],
    }


def _berreman_matrix(epsilon_r: np.ndarray, omega: float) -> np.ndarray:
    """4x4 tangential Maxwell generator for a z-stratified anisotropic medium."""
    eps = np.asarray(epsilon_r, dtype=complex)
    if eps.shape != (3, 3) or not np.all(np.isfinite(eps)):
        raise ValueError("epsilon_r must be a finite complex 3x3 tensor")
    if abs(eps[2, 2]) <= 1e-12 * max(float(np.linalg.norm(eps)), 1.0):
        raise ValueError("epsilon_zz is singular; cannot eliminate longitudinal E")
    ezx, ezy = -eps[2, 0] / eps[2, 2], -eps[2, 1] / eps[2, 2]
    exx, exy = eps[0, 0] + eps[0, 2]*ezx, eps[0, 1] + eps[0, 2]*ezy
    eyx, eyy = eps[1, 0] + eps[1, 2]*ezx, eps[1, 1] + eps[1, 2]*ezy
    a = np.zeros((4, 4), dtype=complex)
    a[0, 3] = 1j * omega * MU0
    a[1, 2] = -1j * omega * MU0
    a[2, 0], a[2, 1] = -1j*omega*EPS0*eyx, -1j*omega*EPS0*eyy
    a[3, 0], a[3, 1] = 1j*omega*EPS0*exx, 1j*omega*EPS0*exy
    return a


def propagate_stratified_vector_maxwell(
    z_m, relative_permittivity: Callable[[float], np.ndarray],
    frequency_Hz: float, initial_tangential_state, *,
    rtol: float = 1e-8, atol: float = 1e-10,
    max_step_m: float = np.inf,
) -> dict:
    """Integrate full non-paraxial vector Maxwell equations along z.

    State ordering is [Ex,Ey,Hx,Hy]. Off-diagonal dielectric elements couple
    polarizations and produce mode conversion. The medium must be invariant
    in x,y and may vary along z. This is an initial-value propagator, not a
    two-sided reflection boundary solver.
    """
    z = np.asarray(z_m, dtype=float)
    y0 = np.asarray(initial_tangential_state, dtype=complex)
    f = float(frequency_Hz)
    if z.ndim != 1 or z.size < 2 or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("z_m must be finite, strictly increasing and have >=2 points")
    if y0.shape != (4,) or not np.all(np.isfinite(y0)):
        raise ValueError("initial state must be finite [Ex,Ey,Hx,Hy]")
    if not np.isfinite(f) or f <= 0 or not callable(relative_permittivity):
        raise ValueError("positive finite frequency and callable permittivity are required")
    if rtol <= 0 or atol <= 0 or max_step_m <= 0:
        raise ValueError("integration tolerances and max_step_m must be positive")
    omega = 2*np.pi*f

    def rhs(position, state):
        return _berreman_matrix(relative_permittivity(float(position)), omega) @ state

    sol = solve_ivp(rhs, (z[0], z[-1]), y0, t_eval=z, method="DOP853",
                    rtol=rtol, atol=atol, max_step=max_step_m)
    if not sol.success or sol.y.shape != (4, z.size) or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"vector Maxwell propagation failed: {sol.message}")
    fields = sol.y.T
    ex, ey, hx, hy = fields.T
    ez = np.empty(z.size, dtype=complex)
    flux = np.empty(z.size)
    cross_fraction = np.empty(z.size)
    for i, position in enumerate(z):
        eps = np.asarray(relative_permittivity(float(position)), dtype=complex)
        if eps.shape != (3, 3) or abs(eps[2, 2]) <= 1e-12*max(float(np.linalg.norm(eps)), 1.0):
            raise ValueError("invalid/singular permittivity during field reconstruction")
        ez[i] = -(eps[2, 0]*ex[i] + eps[2, 1]*ey[i])/eps[2, 2]
        flux[i] = 0.5*np.real(ex[i]*np.conj(hy[i]) - ey[i]*np.conj(hx[i]))
        norm = abs(ex[i])**2 + abs(ey[i])**2
        cross_fraction[i] = abs(ey[i])**2/norm if norm > 0 else np.nan
    return {
        "z_m": z, "electric_field_V_m": np.column_stack((ex, ey, ez)),
        "magnetic_field_A_m": np.column_stack((hx, hy, np.zeros_like(hx))),
        "tangential_state": fields, "longitudinal_poynting_flux_W_m2": flux,
        "cross_polarized_fraction": cross_fraction,
        "solver": "DOP853_full_vector_stratified_Maxwell_Berreman_4x4",
        "nonparaxial": True, "polarization_coupling_included": True,
        "assumptions": [
            "Caller-supplied linear local dielectric tensor.",
            "Medium varies only along z; transverse diffraction/refraction omitted.",
            "Initial-value propagation; reflected-wave boundary matching omitted.",
        ],
        "integration": {"success": bool(sol.success), "message": str(sol.message),
                        "nfev": int(sol.nfev), "rtol": float(rtol), "atol": float(atol)},
    }
