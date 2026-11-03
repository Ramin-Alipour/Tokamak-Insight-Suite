"""Magnetized kinetic ECE absorption from the linear Vlasov conductivity.

Absorption is normalized by the actual incident normal Poynting flux of the
supplied plane-wave eigenmode, rather than assuming vacuum impedance. The
kinetic conductivity is non-relativistic Maxwellian; this is not a relativistic
or collisional-Landau closure.
"""
from __future__ import annotations
import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from .cts_full_em import _species_conductivity, ME, E_CHARGE


def kinetic_ece_absorption_coefficient(
    frequency_Hz, wavevector_rad_m, magnetic_field_T, *,
    electron_density_m3, electron_temperature_eV,
    polarization, harmonic_order=8, quadrature_order=48,
    damping_rate_rad_s=0.0,
):
    """Return kinetic absorption alpha [m^-1] for a supplied EM eigenmode.

    For exp(-i omega t), Q=Re(E* . sigma . E)/2 and
    alpha=Q/<S_n>. The supplied wavevector and polarization must belong to
    the same local mode. Negative alpha (net kinetic gain) is retained.
    """
    f, ne, te = map(float, (frequency_Hz, electron_density_m3, electron_temperature_eV))
    k = np.asarray(wavevector_rad_m, dtype=float)
    b = np.asarray(magnetic_field_T, dtype=float)
    e = np.asarray(polarization, dtype=complex)
    gamma = float(damping_rate_rad_s)
    if not np.isfinite(f) or f <= 0 or not np.isfinite(ne) or ne <= 0 or not np.isfinite(te) or te <= 0:
        raise ValueError("frequency, density and temperature must be positive and finite")
    if k.shape != (3,) or b.shape != (3,) or e.shape != (3,) or any(np.any(~np.isfinite(x)) for x in (k,b,e)):
        raise ValueError("wavevector, magnetic field and polarization must be finite 3-vectors")
    if np.linalg.norm(k) == 0 or np.linalg.norm(b) == 0 or np.linalg.norm(e) == 0:
        raise ValueError("wavevector, magnetic field and polarization must be non-zero")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("damping_rate_rad_s must be finite and non-negative")
    if isinstance(harmonic_order, (bool, np.bool_)) or int(harmonic_order) != harmonic_order or harmonic_order < 1:
        raise ValueError("harmonic_order must be a positive integer")
    if isinstance(quadrature_order, (bool, np.bool_)) or int(quadrature_order) != quadrature_order or quadrature_order < 16:
        raise ValueError("quadrature_order must be an integer >= 16")
    omega = 2*np.pi*f
    species = {"name":"e-", "density":ne, "temperature":te,
               "mass":ME, "charge":-E_CHARGE, "flow":np.zeros(3)}
    sigma = _species_conductivity(
        omega, k, b, species, harmonic_order=int(harmonic_order),
        quadrature_order=int(quadrature_order), damping_rate_rad_s=gamma,
    )
    e = e / np.linalg.norm(e)
    dissipated = float(np.real(np.vdot(e, sigma @ e)))
    # H=k x E/(mu0 omega); the signed normal flux follows the wavevector.
    h = np.cross(k, e)/(mu_0*omega)
    khat = k/np.linalg.norm(k)
    flux = 0.5*float(np.real(np.dot(np.cross(e, np.conj(h)), khat)))
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("supplied polarization/wavevector has non-positive normal Poynting flux")
    alpha = 0.5*dissipated/flux
    return {
        "absorption_coefficient_m_inverse": float(alpha),
        "conductivity_S_m": sigma,
        "polarization_unit": e,
        "dissipative_conductivity_S_m": dissipated,
        "incident_normal_poynting_flux_W_m2_per_V2_m2": flux,
        "frequency_Hz": f,
        "harmonic_order": int(harmonic_order),
        "quadrature_order": int(quadrature_order),
        "damping_rate_rad_s": gamma,
        "model": "magnetized_nonrelativistic_Maxwellian_linear_Vlasov_conductivity",
        "absorption_normalization": "Poynting_theorem_actual_supplied_plane_wave_mode",
        "limitations": [
            "No relativistic/Juttner distribution averaging.",
            "No self-consistent ray tracing or mode conversion; supply the local mode wavevector and polarization.",
            "Damping is user-specified; this routine does not implement a conserving collisional Landau operator.",
            "The plane-wave flux normalization assumes a homogeneous local mode and does not include beam-envelope energy transport.",
        ],
    }
