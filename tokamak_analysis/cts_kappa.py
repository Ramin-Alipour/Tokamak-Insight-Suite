"""Collisionless isotropic-Kappa electrostatic susceptibility.

This is the exact one-dimensional marginal of an isotropic three-dimensional
Kappa distribution. It is deliberately limited to a collisionless,
unmagnetized, longitudinal response; it is not a relativistic or Full-EM
Kappa closure.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, betainc, wofz
from scipy.integrate import quad
from .cts_kappa_dispersion import kappa_dispersion_function


E = 1.602176634e-19
EPS0 = 8.8541878128e-12
ME = 9.1093837015e-31
C = 299792458.0  # exact SI speed of light, m s^-1


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


def _kappa_landau_real_axis_integrals(kappa, normalized_phase_velocity):
    """Exact retarded Landau integral and phase derivative at eta -> 0+."""
    kap, xp = float(kappa), float(normalized_phase_velocity)
    log_norm = gammaln(kap)-gammaln(kap-0.5)-0.5*np.log(np.pi)
    def pdf(x):
        return np.exp(log_norm-kap*np.log1p(x*x))
    def first(x):
        return -2.0*kap*x/(1.0+x*x)*pdf(x)
    def second(x):
        return pdf(x)*(-2.0*kap*(1.0-x*x)+4.0*kap*kap*x*x)/(1.0+x*x)**2
    def third(x):
        lp=-2.0*kap*x/(1.0+x*x)
        lpp=-2.0*kap*(1.0-x*x)/(1.0+x*x)**2
        lppp=4.0*kap*x*(3.0-x*x)/(1.0+x*x)**3
        return pdf(x)*(lp**3+3.0*lp*lpp+lppp)
    fp,fpp,fppp=first(xp),second(xp),third(xp)
    scale=max(1.0,abs(xp))
    def pv_integrand(x):
        d=x-xp
        return fpp+0.5*fppp*d if abs(d)<1e-4*scale else (first(x)-fp)/d
    def derivative_integrand(x):
        d=x-xp
        return 0.5*fppp if abs(d)<2e-4*scale else (first(x)-fp-fpp*d)/(d*d)
    real,real_error=quad(pv_integrand,-np.inf,np.inf,epsabs=2e-11,epsrel=2e-9,limit=500)
    derivative,derivative_error=quad(derivative_integrand,-np.inf,np.inf,epsabs=2e-10,epsrel=2e-8,limit=500)
    return complex(real,np.pi*fp),complex(derivative,np.pi*fpp),float(real_error),float(derivative_error)


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
    electron_flow_projection_m_s=0.0,
):
    """Return chi_e for a drifting normalized 1-D marginal of an isotropic 3-D Kappa.

    The temperature parameter is the usual second-moment temperature, so the
    marginal variance is k_B T / m. A signed projected electron bulk drift
    shifts the resonant phase velocity to v_phase,rest=v_phase,lab-u_e. The
    analytic Kappa CDF quantifies the mass
    represented by the finite grid. Grids truncating too much of the heavy tail
    are rejected rather than silently renormalized into an artificially narrow
    distribution. The default retarded Landau contour is the exact eta-to-zero Sokhotski-Plemelj limit; a finite positive pole displacement is optional and explicit.
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
    if (isinstance(electron_flow_projection_m_s, (bool, np.bool_))
            or not np.isscalar(electron_flow_projection_m_s)):
        raise ValueError("electron_flow_projection_m_s must be a finite subluminal scalar")
    electron_flow = float(electron_flow_projection_m_s)
    if not np.isfinite(electron_flow) or abs(electron_flow) >= C:
        raise ValueError("electron_flow_projection_m_s must be finite and strictly subluminal")
    phase_velocity_lab = phase_velocity
    phase_velocity -= electron_flow
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
    # Use the shared generalized Kappa dispersion kernel as the canonical
    # Landau integral.  Its dimensionless coordinate is
    #   x = v / sqrt(((2*kappa-3)/kappa) * k_B*T/m),
    # whereas the PDF above is written with scale sqrt((2*kappa-3)*k_B*T/m).
    # The two coordinates differ by sqrt(kappa); the integral and its
    # phase-velocity derivative therefore transform with powers 2 and 3.
    # This explicit conversion prevents the optical/CTS Kappa kernels from
    # drifting into inconsistent thermal-speed conventions.
    dispersion_velocity_scale = float(np.sqrt(
        ((2.0*kap-3.0)/kap) * E*te/mass
    ))
    eta = (
        0.0 if pole_imaginary_part_m_s is None
        else _positive_real_scalar(pole_imaginary_part_m_s, "pole_imaginary_part_m_s")
    )
    normalized_phase_velocity = phase_velocity/dispersion_velocity_scale
    normalized_pole_imaginary = eta/dispersion_velocity_scale
    dispersion = kappa_dispersion_function(
        normalized_phase_velocity, normalized_pole_imaginary, kap
    )
    integral = complex(
        dispersion["kappa_dispersion_integral"]/dispersion_velocity_scale**2
    )
    integral_phase_derivative = complex(
        dispersion["kappa_dispersion_integral_derivative"]/dispersion_velocity_scale**3
    )
    integral_error = float(
        dispersion["absolute_quadrature_error_estimate"]/dispersion_velocity_scale**2
    )
    derivative_error = float(
        dispersion["derivative_absolute_quadrature_error_estimate"]/dispersion_velocity_scale**3
    )
    # The shared kernel reports a norm bound for complex quadrature.  It is a
    # conservative component-wise bound for positive eta; on the exact
    # Plemelj boundary the resonant imaginary part is analytic, not quadrature.
    real_error = integral_error
    imag_error = integral_error if eta > 0.0 else 0.0
    derivative_real_error = derivative_error
    derivative_imag_error = derivative_error if eta > 0.0 else 0.0
    zeta = complex(phase_velocity, eta)
    quadrature = dispersion["quadrature_method"]
    contour_method = (
        "exact_eta_to_zero_plus" if eta == 0.0
        else "finite_positive_imaginary_pole_regularization"
    )

    response = {
        "integral": complex(integral),
        "integral_phase_velocity_derivative_s3_m3": complex(integral_phase_derivative),
        "integral_phase_velocity_derivative_real_absolute_error": float(derivative_real_error),
        "integral_phase_velocity_derivative_imag_absolute_error": float(derivative_imag_error),
        "interior_derivative_integral": complex(integral),
        "finite_support_boundary_integral": 0.0j,
        "normalized_probability_density_s_m": pdf.copy(),
        "grid_conditioned_probability_density_s_m": pdf.copy()/grid_coverage,
        "grid_conditioning_applied_to_physical_response": False,
        "analytic_kappa_pdf_s_m": pdf.copy(),
        "pole": zeta,
        "normalized_phase_velocity": complex(normalized_phase_velocity, normalized_pole_imaginary),
        "dispersion_kernel": "shared_full_tail_generalized_kappa_landau_integral",
        "dispersion_velocity_scale_m_s": dispersion_velocity_scale,
        "pole_imaginary_part_m_s": float(eta),
        "pole_order": 1,
        "quadrature": quadrature,
        "landau_contour_method": contour_method,
        "finite_pole_regularization_used": bool(pole_imaginary_part_m_s is not None),
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
        "susceptibility_phase_velocity_derivative_s_m": complex(
            -plasma_frequency_squared / wave_number**2
            * response["integral_phase_velocity_derivative_s3_m3"]
        ),
        "susceptibility_angular_frequency_derivative_s": complex(
            -plasma_frequency_squared / wave_number**3
            * response["integral_phase_velocity_derivative_s3_m3"]
        ),
        "plasma_frequency_squared_s_minus2": float(plasma_frequency_squared),
        "wavenumber_m_inverse": wave_number,
        "phase_velocity_lab_m_s": phase_velocity_lab,
        "phase_velocity_in_electron_rest_frame_m_s": phase_velocity,
        "electron_flow_projection_m_s": electron_flow,
        "distribution_model": "exact_1D_marginal_of_isotropic_3D_Kappa",
        "kappa": kap,
        "theta_m_s": float(np.sqrt(theta_squared)),
        "second_moment_m2_s2": float(E * te / mass),
        "fourth_moment_exists": bool(kap > 2.5),
        "kurtosis": float(3.0 * (2.0 * kap - 3.0) / (2.0 * kap - 5.0)) if kap > 2.5 else None,
        "excess_kurtosis": float(6.0 / (2.0 * kap - 5.0)) if kap > 2.5 else None,
        "fourth_central_moment_m4_s4": (
            float(3.0 * (2.0 * kap - 3.0) / (2.0 * kap - 5.0) * (E * te / mass)**2)
            if kap > 2.5 else None
        ),
        "velocity_grid_probability_coverage": grid_coverage,
        "finite_support_probability_mass": grid_coverage,
        "minimum_velocity_grid_probability_coverage": minimum_coverage,
        "finite_velocity_grid_truncation": bool(grid_coverage < 1.0 - 1e-12),
        "finite_grid_renormalization_bias_prevented": True,
        "finite_grid_normalization_bias_prevented": True,
        "susceptibility_uses_complete_analytic_kappa_tail": True,
        "finite_support_boundary_terms_included": False,
        "tail_model": "exact_analytic_isotropic_Kappa_marginal_integrated_over_full_velocity_axis; finite_grid_used_only_for_probability_coverage_gate",

        "limitations": [
            "Collisionless unmagnetized longitudinal response only.",
            "No conserving collisions, FDT noise, magnetized response or Full-EM vertex.",
            "A 1-D marginal is not a reconstruction of the full 3-D EVDF.",
            "The result is non-relativistic; relativistic Kappa response is not implemented.",
            "The default response is the exact retarded eta-to-zero Landau limit; an explicitly supplied positive pole displacement selects a finite-pole regularized response.",
            "The phase-velocity derivative is analytic in both contour modes; the exact Landau mode includes the i*pi*f'' resonant residue.",
        ],
    })
    return response


def maxwellian_electron_susceptibility(
    phase_velocity_m_s, *, electron_density_m3, electron_temperature_eV,
    wavenumber_m_inverse, mass_kg=ME,
    electron_flow_projection_m_s=0.0,
):
    r"""Exact collisionless Maxwellian longitudinal susceptibility.

    Uses the Fried-Conte plasma dispersion function
    Z(zeta) = i*sqrt(pi)*wofz(zeta), with
    zeta = v_phase/sqrt(2*k_B*T/m), and
    chi_e = (2*omega_pe**2/(k**2*v_th**2)) * (1 + zeta*Z(zeta)).
    The retarded Landau contour is selected by the Faddeeva function for
    real phase velocity. A signed projected electron drift u_e is included
    through zeta=(v_phase_lab-u_e)/v_th, so the resonance is evaluated in the
    electron bulk rest frame. This is an analytic Maxwellian reference kernel,
    not a collisional, magnetized, or electromagnetic CTS model.
    """
    raw_phase = np.asarray(phase_velocity_m_s)
    if raw_phase.dtype.kind not in "iuf" or raw_phase.ndim > 1:
        raise ValueError("phase_velocity_m_s must be a real scalar or 1-D array")
    phase = raw_phase.astype(float, copy=False)
    if np.any(~np.isfinite(phase)):
        raise ValueError("phase_velocity_m_s must be finite")
    if (isinstance(electron_flow_projection_m_s, (bool, np.bool_))
            or not np.isscalar(electron_flow_projection_m_s)):
        raise ValueError("electron_flow_projection_m_s must be a finite subluminal scalar")
    electron_flow = float(electron_flow_projection_m_s)
    if not np.isfinite(electron_flow) or abs(electron_flow) >= C:
        raise ValueError("electron_flow_projection_m_s must be finite and strictly subluminal")
    relative_phase = phase - electron_flow
    ne = _positive_real_scalar(electron_density_m3, "electron_density_m3")
    te = _positive_real_scalar(electron_temperature_eV, "electron_temperature_eV")
    wave_number = _positive_real_scalar(wavenumber_m_inverse, "wavenumber_m_inverse")
    mass = _positive_real_scalar(mass_kg, "mass_kg")
    sigma2 = E * te / mass
    thermal_speed = float(np.sqrt(2.0 * sigma2))
    omega_pe2 = ne * E**2 / (EPS0 * mass)
    zeta = relative_phase / thermal_speed
    plasma_dispersion = 1j * np.sqrt(np.pi) * wofz(zeta)
    susceptibility_scale = omega_pe2 / (wave_number**2 * sigma2)
    susceptibility = susceptibility_scale * (1.0 + zeta * plasma_dispersion)
    # Fried-Conte identity: Z'(zeta) = -2 [1 + zeta Z(zeta)].
    # This is the analytic phase-velocity sensitivity, not a finite difference.
    plasma_dispersion_derivative = -2.0 * (1.0 + zeta * plasma_dispersion)
    susceptibility_phase_derivative = (
        susceptibility_scale / thermal_speed
        * (plasma_dispersion + zeta * plasma_dispersion_derivative)
    )
    if np.any(~np.isfinite(susceptibility.real)) or np.any(~np.isfinite(susceptibility.imag)):
        raise FloatingPointError("analytic Maxwellian susceptibility is non-finite")
    scalar_input = raw_phase.ndim == 0
    return {
        "electron_susceptibility": complex(susceptibility) if scalar_input else susceptibility,
        "plasma_dispersion_function": complex(plasma_dispersion) if scalar_input else plasma_dispersion,
        "plasma_dispersion_function_derivative": complex(plasma_dispersion_derivative) if scalar_input else plasma_dispersion_derivative,
        "susceptibility_phase_velocity_derivative_s_m": complex(susceptibility_phase_derivative) if scalar_input else susceptibility_phase_derivative,
        "normalized_phase_velocity": float(zeta) if scalar_input else zeta,
        "phase_velocity_lab_m_s": float(phase) if scalar_input else phase,
        "phase_velocity_in_electron_rest_frame_m_s": float(relative_phase) if scalar_input else relative_phase,
        "electron_flow_projection_m_s": electron_flow,
        "thermal_speed_m_s": thermal_speed,
        "one_dimensional_velocity_variance_m2_s2": float(sigma2),
        "plasma_frequency_squared_s_minus2": float(omega_pe2),
        "wavenumber_m_inverse": wave_number,
        "distribution_model": "isotropic_nonrelativistic_Maxwellian",
        "landau_contour": "retarded_Fried_Conte_Z_i_sqrt_pi_wofz",
        "response_model": "collisionless_unmagnetized_longitudinal_electrostatic",
        "limitations": [
            "No collisional, magnetized, relativistic, or Full-EM response.",
            "No ion contribution or fluctuation-dissipation spectrum; this is electron susceptibility only.",
        ],
    }


def bi_kappa_electron_susceptibility(
    v, phase_velocity_m_s, *, electron_density_m3,
    electron_temperature_parallel_eV, electron_temperature_perpendicular_eV,
    magnetic_field_unit, scattering_vector_unit, kappa,
    wavenumber_m_inverse, mass_kg=ME,
    pole_imaginary_part_m_s=None,
    minimum_velocity_grid_probability_coverage=0.999,
    electron_flow_projection_m_s=0.0,
):
    r"""Collisionless longitudinal susceptibility of a drifting gyrotropic bi-Kappa EVDF.

    The 3-D distribution is proportional to
      [1 + v_perp^2/(kappa theta_perp^2)
         + v_parallel^2/(kappa theta_parallel^2)]^(-(kappa+1)).
    Integrating over the two velocity coordinates perpendicular to the
    scattering wavevector gives an exact 1-D Kappa marginal with directional
    second-moment temperature
      T_q = T_perp + (T_parallel-T_perp) (b_hat.q_hat)^2.
    The Landau integral is then evaluated by the existing full-tail analytic
    Kappa solver. This is a gyrotropic, non-relativistic, unmagnetized
    longitudinal projection; it is not a magnetized harmonic or Full-EM
    susceptibility. A field-aligned or oblique bulk drift enters through the
    Galilean-shifted resonant phase velocity v_phi - u_e dot q_hat. One view
    measures T_q, not both temperatures separately.
    """
    b = np.asarray(magnetic_field_unit)
    q = np.asarray(scattering_vector_unit)
    if (b.dtype.kind not in "iuf" or q.dtype.kind not in "iuf"
            or b.shape != (3,) or q.shape != (3,)):
        raise ValueError("magnetic_field_unit and scattering_vector_unit must be real three-component vectors")
    b = b.astype(float, copy=False)
    q = q.astype(float, copy=False)
    if (not np.all(np.isfinite(b)) or not np.all(np.isfinite(q))
            or not np.isclose(np.linalg.norm(b), 1.0, rtol=1e-8, atol=1e-10)
            or not np.isclose(np.linalg.norm(q), 1.0, rtol=1e-8, atol=1e-10)):
        raise ValueError("magnetic_field_unit and scattering_vector_unit must be finite unit vectors")
    t_parallel = _positive_real_scalar(electron_temperature_parallel_eV, "electron_temperature_parallel_eV")
    t_perpendicular = _positive_real_scalar(electron_temperature_perpendicular_eV, "electron_temperature_perpendicular_eV")
    if (isinstance(electron_flow_projection_m_s, (bool, np.bool_))
            or not np.isscalar(electron_flow_projection_m_s)):
        raise ValueError("electron_flow_projection_m_s must be a finite subluminal scalar")
    electron_flow = float(electron_flow_projection_m_s)
    if not np.isfinite(electron_flow) or abs(electron_flow) >= C:
        raise ValueError("electron_flow_projection_m_s must be finite and strictly subluminal")
    phase_velocity = float(phase_velocity_m_s)
    if not np.isfinite(phase_velocity):
        raise ValueError("phase_velocity_m_s must be finite")
    relative_phase_velocity = phase_velocity - electron_flow
    mu = float(np.clip(np.dot(b, q), -1.0, 1.0))
    t_projected = t_perpendicular + (t_parallel - t_perpendicular) * mu**2
    result = kappa_electron_susceptibility(
        v, relative_phase_velocity,
        electron_density_m3=electron_density_m3,
        electron_temperature_eV=t_projected,
        kappa=kappa, wavenumber_m_inverse=wavenumber_m_inverse,
        mass_kg=mass_kg,
        pole_imaginary_part_m_s=pole_imaginary_part_m_s,
        minimum_velocity_grid_probability_coverage=minimum_velocity_grid_probability_coverage,
    )
    result.update({
        "distribution_model": "gyrotropic_3D_bi_Kappa_exact_1D_directional_marginal",
        "electron_temperature_parallel_eV": t_parallel,
        "electron_temperature_perpendicular_eV": t_perpendicular,
        "projected_temperature_eV": float(t_projected),
        "electron_flow_projection_m_s": electron_flow,
        "phase_velocity_lab_m_s": phase_velocity,
        "phase_velocity_in_electron_rest_frame_m_s": float(relative_phase_velocity),
        "resonance_frame": "electron_bulk_rest_frame_using_v_phase_minus_u_e_dot_q_hat",
        "magnetic_pitch_cosine": mu,
        "directional_temperature_relation": "T_q = T_perp + (T_parallel - T_perp)*(b_hat dot q_hat)^2",
        "separate_parallel_perpendicular_temperatures_identifiable_from_one_view": False,
        "anisotropy_model": "gyrotropic_bi_Kappa_with_common_kappa_and_projected_bulk_drift",
        "limitations": [
            "Collisionless unmagnetized longitudinal response only; no cyclotron harmonics or magnetic-field-dependent dielectric tensor.",
            "Non-relativistic distribution; no relativistic bi-Kappa response.",
            "One scattering direction identifies only the directional temperature combination T_q.",
            "No ion susceptibility, fluctuation-dissipation source, collisions, or Full-EM vertex is included.",
        ],
    })
    return result
