"""Full-tail generalized Kappa plasma dispersion integral.

The normalized 1-D marginal is the exact projection of an isotropic 3-D
Kappa EVDF.  The returned dispersion integral is
    Z'_kappa(zeta) = integral_{-inf}^{inf} f'_kappa(x)/(x-zeta) dx,
with the retarded Landau boundary value at Im(zeta)=0+.
This is a longitudinal, unmagnetized response kernel; it is not a magnetized
harmonic dielectric or a fluctuation-dissipation closure.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln


def kappa_dispersion_function(
    zeta_real,
    zeta_imag,
    kappa,
    *,
    quadrature_epsabs=1.0e-10,
    quadrature_epsrel=1.0e-9,
):
    """Evaluate the full-tail Kappa Landau dispersion integral and derivative.

    The dimensionless velocity is x=v/sqrt(((2*kappa-3)/kappa)*k_B*T/m).
    The normalized marginal has Var(x)=kappa/(2*kappa-3), so this scaling
    gives Var(v)=k_B*T/m and tends to the Maxwellian thermal scale
    sqrt(2*k_B*T/m) as kappa->infinity.  zeta_imag must be non-negative. At zero,
    the real part is evaluated as a Cauchy principal value and the imaginary
    part is supplied by Sokhotski-Plemelj: +pi*f'(zeta_real).

    For positive imaginary part, both the integral and its analytic derivative
    d/dzeta integral f'(x)/(x-zeta) dx are integrated over the complete tails.
    """
    values = (zeta_real, zeta_imag, kappa, quadrature_epsabs, quadrature_epsrel)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(
        v, (int, float, np.integer, np.floating)
    ) for v in values):
        raise ValueError("zeta, kappa and quadrature tolerances must be real scalars")
    xr, yi, kap, epsabs, epsrel = map(float, values)
    if not np.all(np.isfinite([xr, yi, kap, epsabs, epsrel])):
        raise ValueError("all inputs must be finite")
    if yi < 0.0:
        raise ValueError("zeta_imag must be non-negative for the retarded Landau branch")
    if kap <= 1.5:
        raise ValueError("kappa must exceed 3/2 for a finite second-moment temperature")
    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("quadrature tolerances must be positive")

    log_norm = gammaln(kap) - gammaln(kap - 0.5) - 0.5*np.log(np.pi*kap)

    def pdf(x):
        return np.exp(log_norm - kap*np.log1p((x*x)/kap))

    def derivative_pdf(x):
        return (-2.0*x/(1.0 + x*x/kap))*pdf(x)

    def second_derivative_pdf(x):
        denominator = 1.0 + x*x/kap
        logarithmic_slope = -2.0*x/denominator
        slope_derivative = -2.0*(1.0 - x*x/kap)/(denominator*denominator)
        return (slope_derivative + logarithmic_slope*logarithmic_slope)*pdf(x)

    def principal_value_over_real_line(function, pole):
        # Map x in (-inf, inf) to t=atan(x) in (-pi/2, pi/2).
        # The transformed Cauchy weight is finite-interval, as required by
        # QUADPACK's weighted principal-value integrator.
        pole_angle = float(np.arctan(pole))

        def transformed_numerator(angle):
            if abs(angle - pole_angle) < 1.0e-7:
                return float(function(pole))
            x = float(np.tan(angle))
            return float(
                function(x) * (1.0 + x*x)
                * (angle - pole_angle) / (x - pole)
            )

        return quad(
            transformed_numerator, -0.5*np.pi, 0.5*np.pi,
            weight="cauchy", wvar=pole_angle,
            epsabs=epsabs, epsrel=epsrel, limit=500,
        )

    zeta = complex(xr, yi)
    if yi == 0.0:
        real, real_error = principal_value_over_real_line(derivative_pdf, xr)
        integral = complex(real, np.pi*derivative_pdf(xr))
        # Integration by parts converts the finite-part second-order pole
        # integral into a first-order Cauchy principal value of f'':
        # d/dz PV int f'(x)/(x-z) dx = PV int f''(x)/(x-z) dx.
        derivative_real, derivative_real_error = principal_value_over_real_line(
            second_derivative_pdf, xr
        )
        derivative = complex(derivative_real, np.pi*second_derivative_pdf(xr))
        derivative_error = float(derivative_real_error)
        method = "full_tail_cauchy_pv_and_analytic_derivative_plus_sokhotski_plemelj"
    else:
        # Resolve the narrow Landau pole explicitly. Direct adaptive integration
        # over the entire real line can miss a sharply peaked resonance when
        # Im(zeta) is small. Map the central +/-50 eta interval to t in [-50,50]
        # and integrate the analytic tails separately.
        def outer_integrand(x):
            return derivative_pdf(x)/(complex(x, 0.0)-zeta)

        def central_integrand(t):
            x = xr + yi*t
            return derivative_pdf(x)/complex(t, -1.0)

        def outer_derivative_integrand(x):
            return derivative_pdf(x)/(complex(x, 0.0)-zeta)**2

        def central_derivative_integrand(t):
            x = xr + yi*t
            return derivative_pdf(x)/(yi*complex(t, -1.0)**2)

        def integrate_component(component, central, outer):
            centre, centre_error = quad(
                lambda t: float(component(central(t))), -50.0, 50.0,
                epsabs=epsabs, epsrel=epsrel, limit=500,
            )
            left, left_error = quad(
                lambda x: float(component(outer(x))), -np.inf, xr-50.0*yi,
                epsabs=epsabs, epsrel=epsrel, limit=500,
            )
            right, right_error = quad(
                lambda x: float(component(outer(x))), xr+50.0*yi, np.inf,
                epsabs=epsabs, epsrel=epsrel, limit=500,
            )
            return centre+left+right, centre_error+left_error+right_error

        integral_real, integral_real_error = integrate_component(
            np.real, central_integrand, outer_integrand
        )
        integral_imag, integral_imag_error = integrate_component(
            np.imag, central_integrand, outer_integrand
        )
        integral = complex(integral_real, integral_imag)

        derivative_real, derivative_real_error = integrate_component(
            np.real, central_derivative_integrand, outer_derivative_integrand
        )
        derivative_imag, derivative_imag_error = integrate_component(
            np.imag, central_derivative_integrand, outer_derivative_integrand
        )
        derivative = complex(derivative_real, derivative_imag)
        derivative_error = float(np.hypot(
            derivative_real_error, derivative_imag_error
        ))
        real_error = float(np.hypot(integral_real_error, integral_imag_error))
        method = "full_tail_pole_resolved_complex_quadrature_upper_half_plane"

    if not np.isfinite(integral.real) or not np.isfinite(integral.imag):
        raise ArithmeticError("Kappa dispersion integral is non-finite")
    return {
        "kappa_dispersion_integral": integral,
        "kappa_dispersion_integral_derivative": derivative,
        "zeta": zeta,
        "kappa": kap,
        "normalization": float(np.exp(log_norm)),
        "quadrature_method": method,
        "absolute_quadrature_error_estimate": float(real_error),
        "derivative_absolute_quadrature_error_estimate": derivative_error,
        "tail_treatment": "analytic_distribution_integrated_over_minus_infinity_to_plus_infinity",
        "landau_branch": "retarded_upper_half_plane_and_exact_sokhotski_plemelj_boundary",
        "maxwellian_limit": "f(x)=exp(-x^2)/sqrt(pi); integral=-2*(1+zeta*Z(zeta))",
        "validity_domain": "isotropic_nonrelativistic_unmagnetized_collisionless_longitudinal_response",
        "limitations": [
            "No magnetized cyclotron harmonics or transverse dielectric response.",
            "No collisional operator or fluctuation-dissipation noise closure.",
            "No relativistic Kappa distribution.",
        ],
    }
