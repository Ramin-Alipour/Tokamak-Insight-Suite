"""Collisionless isotropic-Kappa electrostatic susceptibility.

This is the exact one-dimensional marginal of an isotropic three-dimensional
Kappa distribution. It is deliberately limited to a collisionless,
unmagnetized, longitudinal response; it is not a relativistic or Full-EM
Kappa closure.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, betainc
from scipy.integrate import quad


E = 1.602176634e-19
EPS0 = 8.8541878128e-12
ME = 9.1093837015e-31


def _positive_real_scalar(value, name):
    """Reject bool/string/complex coercions before validating a physical scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a positive finite real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return result


def kappa_electron_susceptibility(
    v,
    phase_velocity_m_s,
    *,
    electron_density_m3,
    electron_temperature_eV,
    kappa,
    wavenumber_m_inverse,
    mass_kg=ME,
    pole_imaginary_part_m_s=None,
    minimum_velocity_grid_probability_coverage=0.999,
):
    """Return chi_e for the normalized 1-D marginal of an isotropic 3-D Kappa.

    The temperature parameter is the usual second-moment temperature, so the
    marginal variance is k_B T / m. The analytic Kappa CDF quantifies the mass
    represented by the finite grid. Grids truncating too much of the heavy tail
    are rejected rather than silently renormalized into an artificially narrow
    distribution. The retarded Landau contour uses a positive imaginary pole.
    """
    raw_v = np.asarray(v)
    if raw_v.dtype.kind not in "iuf":
        raise ValueError("v must contain real numeric values")
    velocity = raw_v.astype(float, copy=False)
    ne = _positive_real_scalar(electron_density_m3, "electron_density_m3")
    te = _positive_real_scalar(electron_temperature_eV, "electron_temperature_eV")
    kap = _positive_real_scalar(kappa, "kappa")
    wave_number = _positive_real_scalar(wavenumber_m_inverse, "wavenumber_m_inverse")
    mass = _positive_real_scalar(mass_kg, "mass_kg")
    if (velocity.ndim != 1 or velocity.size < 9
            or np.any(~np.isfinite(velocity)) or np.any(np.diff(velocity) <= 0.0)):
        raise ValueError("velocity grid must be finite, increasing, and contain at least 9 points")
    if kap <= 1.5:
        raise ValueError("kappa must be greater than 3/2")
    if isinstance(phase_velocity_m_s, (bool, np.bool_)) or not isinstance(
        phase_velocity_m_s, (int, float, np.integer, np.floating)
    ):
        raise ValueError("phase_velocity_m_s must be a finite real scalar")
    phase_velocity = float(phase_velocity_m_s)
    if not np.isfinite(phase_velocity):
        raise ValueError("phase_velocity_m_s must be a finite real scalar")
    if pole_imaginary_part_m_s is not None:
        _positive_real_scalar(pole_imaginary_part_m_s, "pole_imaginary_part_m_s")
    if (isinstance(minimum_velocity_grid_probability_coverage, (bool, np.bool_))
            or not isinstance(minimum_velocity_grid_probability_coverage,
                              (int, float, np.integer, np.floating))):
        raise ValueError("minimum_velocity_grid_probability_coverage must be in (0, 1]")
    minimum_coverage = float(minimum_velocity_grid_probability_coverage)
    if not np.isfinite(minimum_coverage) or not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_velocity_grid_probability_coverage must be in (0, 1]")

    # For an isotropic 3-D Kappa, integrating over the two perpendicular
    # velocity coordinates gives f_1D(v) proportional to
    # [1 + v^2/(kappa theta^2)]^(-kappa), with
    # theta^2 = (2 kappa - 3) k_B T / (kappa m).
    theta_squared = (2.0 * kap - 3.0) * E * te / (kap * mass)
    scale_squared = kap * theta_squared
    log_normalization = (
        gammaln(kap) - 0.5 * np.log(np.pi * scale_squared)
        - gammaln(kap - 0.5)
    )
    log_pdf = log_normalization - kap * np.log1p(
        velocity * velocity / scale_squared
    )
    pdf = np.exp(log_pdf)
    # Exact CDF of the normalized 1-D Kappa marginal. This avoids estimating
    # heavy-tail coverage from a coarse-grid trapezoid integral.
    cdf_argument = velocity**2 / (velocity**2 + scale_squared)
    cdf = 0.5 + 0.5*np.sign(velocity)*betainc(0.5, kap-0.5, cdf_argument)
    grid_coverage = float(cdf[-1] - cdf[0])
    if grid_coverage < minimum_coverage:
        raise ValueError(
            "velocity grid captures only "
            f"{grid_coverage:.8g} of the analytic Kappa probability mass; "
            f"minimum_velocity_grid_probability_coverage={minimum_coverage:.8g}. "
            "Expand the velocity grid or explicitly lower the coverage threshold."
        )
    # Evaluate the Landau integral over the complete analytic Kappa tail.
    # A finite-grid derivative integral omits the non-zero derivative outside
    # the tabulated interval; merely multiplying by the represented probability
    # mass does not restore that missing susceptibility.  Integrate the exact
    # marginal derivative on (-inf, inf), keeping the positive retarded pole.
    eta = (
        1e-6 * float(np.median(np.diff(velocity)))
        if pole_imaginary_part_m_s is None
        else float(pole_imaginary_part_m_s)
    )
    zeta = complex(phase_velocity, eta)
    def landau_integrand(real_velocity):
        pdf_value = np.exp(
            log_normalization - kap*np.log1p(real_velocity**2/scale_squared)
        )
        derivative = (
            -2.0*kap*real_velocity/(scale_squared+real_velocity**2)
        )*pdf_value
        return derivative/(complex(real_velocity, 0.0)-zeta)

    integral_real, real_error = quad(
        lambda value: float(np.real(landau_integrand(value))),
        -np.inf, np.inf, epsabs=1e-11, epsrel=2e-8, limit=500,
    )
    integral_imag, imag_error = quad(
        lambda value: float(np.imag(landau_integrand(value))),
        -np.inf, np.inf, epsabs=1e-11, epsrel=2e-8, limit=500,
    )
    integral = complex(integral_real, integral_imag)
    if not np.isfinite(integral.real) or not np.isfinite(integral.imag):
        raise ValueError("full-tail Kappa Landau integral did not converge to a finite value")
    response = {
        "integral": integral,
        "interior_derivative_integral": integral,
        "finite_support_boundary_integral": 0.0j,
        "normalized_probability_density_s_m": pdf.copy() / grid_coverage,
        "pole": zeta,
        "pole_imaginary_part_m_s": eta,
        "pole_order": 1,
        "quadrature": "adaptive_complex_quadrature_of_exact_analytic_Kappa_derivative_over_minus_infinity_to_infinity",
        "finite_support_extension": "analytic_Kappa_tail_integrated_to_infinity",
        "finite_support_boundary_terms_included": False,
        "landau_integral_real_absolute_error": float(real_error),
        "landau_integral_imag_absolute_error": float(imag_error),
        "landau_integral_tail_complete": True,
    }
    plasma_frequency_squared = ne * E**2 / (EPS0 * mass)
    response.update({
        "electron_susceptibility": complex(
            -plasma_frequency_squared / wave_number**2 * response["integral"]
        ),
        "plasma_frequency_squared_s_minus2": float(plasma_frequency_squared),
        "wavenumber_m_inverse": wave_number,
        "distribution_model": "exact_1D_marginal_of_isotropic_3D_Kappa",
        "kappa": kap,
        "theta_m_s": float(np.sqrt(theta_squared)),
        "second_moment_m2_s2": float(E * te / mass),
        "velocity_grid_probability_coverage": grid_coverage,
        "minimum_velocity_grid_probability_coverage": minimum_coverage,
        "finite_velocity_grid_truncation": bool(grid_coverage < 1.0 - 1e-12),
        "finite_grid_renormalization_bias_prevented": True,
        "finite_grid_normalization_bias_prevented": True,
        "susceptibility_uses_complete_analytic_kappa_tail": True,
        "finite_support_boundary_terms_included": False,
        "tail_model": "analytic_isotropic_Kappa_marginal_truncated_only_for_numerical_quadrature; no artificial zero-extension edge jumps",

        "limitations": [
            "Collisionless unmagnetized longitudinal response only.",
            "No conserving collisions, FDT noise, magnetized response or Full-EM vertex.",
            "A 1-D marginal is not a reconstruction of the full 3-D EVDF.",
            "The result is non-relativistic; relativistic Kappa response is not implemented.",
            "Finite velocity support and positive Landau-pole displacement affect numerical accuracy.",
        ],
    })
    return response
