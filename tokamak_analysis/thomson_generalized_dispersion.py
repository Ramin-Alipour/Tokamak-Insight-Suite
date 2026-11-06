"""Generalized unmagnetized electrostatic CTS response for tabulated 1-D EVDFs.

The susceptibility integral uses exact interval integration of the derivative
of a piecewise-linear projected VDF. The primitive supports one complex pole
of any positive integer order per call; the CTS dielectric uses the physical
first-order Landau pole. It is Skolar-inspired and does not implement a
multiple-pole summation or the paper's full kinetic closure.
"""
from __future__ import annotations

import numpy as np


def _trapezoid(y, x):
    """NumPy-version-independent trapezoidal integration."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return integrate(y, x)

from .thomson_kinetic_models import (
    C, E_CHARGE, ME, EPS0, _resolve_ion_species,
    _maxwellian_pdf_and_derivatives, _landau_susceptibility,
)


def generalized_landau_integral(velocity_grid_m_s, probability_density_s_m,
                                phase_velocity_m_s, *, pole_imaginary_part_m_s=None,
                                pole_order=1, include_finite_support_boundary_terms=True, landau_limit=False):
    """Evaluate integral f'(v)/(v-zeta) dv for a tabulated projected VDF.

    f is linearly interpolated on the supplied velocity grid, so f' is
    piecewise constant and each interval integral is analytic for any positive
    integer pole_order. A positive imaginary pole displacement selects the
    retarded/Landau branch and avoids singular real-axis evaluation.
    """
    v = np.asarray(velocity_grid_m_s, dtype=float)
    f = np.asarray(probability_density_s_m, dtype=float)
    vp = float(phase_velocity_m_s)
    if v.ndim != 1 or f.shape != v.shape or v.size < 9:
        raise ValueError("velocity grid and VDF must be matching 1-D arrays with at least 9 points")
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(f)) or np.any(np.diff(v) <= 0) or np.any(f < 0):
        raise ValueError("velocity grid must increase strictly and VDF must be finite and non-negative")
    if not np.isfinite(vp):
        raise ValueError("phase_velocity_m_s must be finite")
    area = float(_trapezoid(f, v))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("tabulated VDF must have positive finite integral")
    f = f / area
    if isinstance(pole_order, (bool, np.bool_)) or int(pole_order) != pole_order or int(pole_order) < 1:
        raise ValueError("pole_order must be a positive integer")
    pole_order = int(pole_order)
    if not isinstance(include_finite_support_boundary_terms, (bool, np.bool_)):
        raise ValueError("include_finite_support_boundary_terms must be boolean")
    if not isinstance(landau_limit, (bool, np.bool_)):
        raise ValueError("landau_limit must be boolean")
    slopes = np.diff(f) / np.diff(v)
    if landau_limit:
        if pole_order != 1:
            raise ValueError("exact Landau limit is implemented only for the first-order pole")
        if pole_imaginary_part_m_s not in (None, 0, 0.0):
            raise ValueError("pole_imaginary_part_m_s must be omitted or zero with landau_limit=True")
        scale_v = max(float(np.max(np.abs(v))), abs(vp), 1.0)
        if np.any(np.abs(v - vp) <= 32.0*np.finfo(float).eps*scale_v):
            raise ValueError("exact Landau limit is undefined at a tabulated derivative knot or support endpoint")
        real_part = 0.0
        for left, right, slope in zip(v[:-1], v[1:], slopes):
            real_part += float(slope)*(np.log(abs(float(right)-vp))-np.log(abs(float(left)-vp)))
        inside = bool(v[0] < vp < v[-1])
        interval = int(np.searchsorted(v, vp)-1) if inside else None
        integral = complex(real_part, np.pi*float(slopes[interval]) if inside else 0.0)
        zeta, eta = complex(vp, 0.0), 0.0
        quadrature = "analytic_principal_value_plus_sokhotski_plemelj_resonance"
    else:
        eta = (1e-6*float(np.median(np.diff(v))) if pole_imaginary_part_m_s is None
               else float(pole_imaginary_part_m_s))
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("pole_imaginary_part_m_s must be positive and finite")
        zeta = complex(vp, eta)
        integral = 0.0j
        quadrature = "analytic_piecewise_linear_derivative_integral"
        for left, right, slope in zip(v[:-1], v[1:], slopes):
            left_delta, right_delta = complex(left, 0.0)-zeta, complex(right, 0.0)-zeta
            if pole_order == 1:
                interval_integral = np.log(right_delta)-np.log(left_delta)
            else:
                interval_integral = (right_delta**(1-pole_order)-left_delta**(1-pole_order))/(1-pole_order)
            integral += float(slope)*interval_integral

    # The tabulated VDF is interpreted as zero outside its finite support.
    # Its distributional derivative therefore contains endpoint jumps:
    # +f(v_min) delta(v-v_min) - f(v_max) delta(v-v_max).  These terms are
    # required by the Landau susceptibility integral whenever a supplied
    # finite-grid VDF does not decay to zero at both endpoints.  Omitting them
    # makes even a top-hat distribution appear to have zero susceptibility.
    boundary_integral = (
        float(f[0]) / (complex(v[0], 0.0) - zeta) ** pole_order
        - float(f[-1]) / (complex(v[-1], 0.0) - zeta) ** pole_order
    ) if include_finite_support_boundary_terms else 0.0j
    integral += boundary_integral
    if not np.isfinite(integral.real) or not np.isfinite(integral.imag):
        raise ValueError("generalized Landau integral is non-finite")
    return {
        "integral": complex(integral),
        "interior_derivative_integral": complex(integral - boundary_integral),
        "finite_support_boundary_integral": complex(boundary_integral),
        "normalized_probability_density_s_m": f,
        "pole": zeta,
        "pole_imaginary_part_m_s": eta,
        "quadrature": quadrature + "_plus_distributional_finite_support_endpoint_jumps",
        "pole_order": pole_order,
        "finite_support_extension": (
            "zero_outside_tabulated_velocity_interval_with_distributional_endpoint_jumps"
            if include_finite_support_boundary_terms else
            "finite_interval_truncation_of_an_analytic_nonzero_tail; endpoint_jumps_omitted"
        ),
        "finite_support_boundary_terms_included": bool(include_finite_support_boundary_terms),
        "landau_limit": bool(landau_limit),
        "limitation": "exact first-order Landau boundary value; derivative knots rejected" if landau_limit else "one complex pole per call; finite velocity support and positive pole displacement are explicit numerical inputs",
    }


def generalized_electron_susceptibility(
    velocity_grid_m_s, probability_density_s_m, phase_velocity_m_s, *,
    electron_density_m3, wavenumber_m_inverse, pole_imaginary_part_m_s=None,
    include_phase_velocity_derivative=False, landau_limit=False,
):
    """Return longitudinal susceptibility and optionally its analytic pole derivative.

    The derivative is evaluated from the second-order Landau kernel at fixed
    positive pole displacement: d chi/d v_phase = -(omega_pe^2/k^2)
    integral f'(v)/(v-v_phase-i eta)^2 dv. It is not a finite difference.
    """
    ne = float(electron_density_m3)
    k = float(wavenumber_m_inverse)
    if not np.isfinite(ne) or ne <= 0 or not np.isfinite(k) or k <= 0:
        raise ValueError("electron density and wavenumber must be positive and finite")
    if not isinstance(include_phase_velocity_derivative, (bool, np.bool_)):
        raise ValueError("include_phase_velocity_derivative must be boolean")
    if not isinstance(landau_limit, (bool, np.bool_)):
        raise ValueError("landau_limit must be boolean")
    if landau_limit and include_phase_velocity_derivative:
        raise ValueError("exact first-order Landau limit does not expose the second-order pole derivative")
    response = generalized_landau_integral(
        velocity_grid_m_s, probability_density_s_m, phase_velocity_m_s,
        pole_imaginary_part_m_s=pole_imaginary_part_m_s, landau_limit=landau_limit,
    )
    omega_pe2 = ne * E_CHARGE**2 / (EPS0 * ME)
    chi = -(omega_pe2 / k**2) * response["integral"]
    response["electron_susceptibility"] = complex(chi)
    response["plasma_frequency_squared_s_minus2"] = float(omega_pe2)
    response["wavenumber_m_inverse"] = k
    if include_phase_velocity_derivative:
        derivative_response = generalized_landau_integral(
            velocity_grid_m_s, probability_density_s_m,
            phase_velocity_m_s,
            pole_imaginary_part_m_s=response["pole_imaginary_part_m_s"],
            pole_order=2,
        )
        derivative_integral = derivative_response["integral"]
        susceptibility_derivative = -(omega_pe2/k**2)*derivative_integral
        response["integral_phase_velocity_derivative_s3_m3"] = derivative_integral
        response["susceptibility_phase_velocity_derivative_s_m"] = complex(susceptibility_derivative)
        response["susceptibility_angular_frequency_derivative_s"] = complex(susceptibility_derivative/k)
        response["phase_derivative_quadrature"] = derivative_response["quadrature"]
        response["phase_derivative_uses_same_fixed_pole_displacement"] = True
    else:
        response["susceptibility_phase_velocity_derivative_s_m"] = None
        response["susceptibility_angular_frequency_derivative_s"] = None
    return response


def collective_tabulated_electron_spectral_density(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
    electron_density_m3, electron_velocity_grid_m_s,
    electron_probability_density_s_m, ion_species, ion_fractions,
    ion_temperature_eV, electron_velocity_m_s, ion_velocity_m_s,
    incident_direction, scattered_direction, pole_imaginary_part_m_s=None,
    landau_limit=False,
):
    """Unmagnetized electrostatic CTS forward spectrum for a supplied projected EVDF.

    The electron distribution is a researcher-supplied 1-D marginal along q.
    Ions remain Maxwellian. This is a forward model, not a generic EVDF inverse;
    no 3-D anisotropy or unconstrained VDF parameters are inferred.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    ne = float(electron_density_m3)
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    vgrid = np.asarray(electron_velocity_grid_m_s, dtype=float)
    fgrid = np.asarray(electron_probability_density_s_m, dtype=float)
    if x.ndim != 1 or x.size < 9 or np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be positive, finite and strictly increasing")
    if not np.isfinite(ne) or ne <= 0 or not np.isfinite(laser) or laser <= 0 or not 0 < angle <= np.pi:
        raise ValueError("density, laser wavelength and scattering angle are invalid")
    ni = np.asarray(incident_direction, dtype=float)
    ns = np.asarray(scattered_direction, dtype=float)
    if ni.shape != (3,) or ns.shape != (3,) or np.any(~np.isfinite(ni)) or np.any(~np.isfinite(ns)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(ni) == 0 or np.linalg.norm(ns) == 0:
        raise ValueError("direction vectors must be non-zero")
    ni, ns = ni/np.linalg.norm(ni), ns/np.linalg.norm(ns)
    if not np.isclose(np.arccos(np.clip(np.dot(ni, ns), -1, 1)), angle, atol=1e-5, rtol=0):
        raise ValueError("scattering_angle_rad is inconsistent with direction vectors")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    ue_vec = np.asarray(electron_velocity_m_s, dtype=float)
    if ue_vec.shape != (3,) or np.any(~np.isfinite(ue_vec)) or np.linalg.norm(ue_vec) >= C:
        raise ValueError("electron_velocity_m_s must be a finite subluminal 3-vector")
    ue = float(np.dot(ue_vec, qhat))
    species = _resolve_ion_species(ion_species)
    fractions = np.asarray(ion_fractions, dtype=float)
    temperatures = np.asarray(np.atleast_1d(ion_temperature_eV), dtype=float)
    vi = np.asarray(ion_velocity_m_s, dtype=float)
    if fractions.shape != (len(species),) or temperatures.shape != (len(species),):
        raise ValueError("ion_fractions and ion_temperature_eV must provide one value per species")
    if (np.any(~np.isfinite(fractions)) or np.any(~np.isfinite(temperatures))
            or np.any(fractions <= 0)
            or not np.isclose(np.sum(fractions), 1.0, rtol=1e-6, atol=1e-8)
            or np.any(temperatures <= 0)):
        raise ValueError("ion fractions must be finite, positive and sum to one; ion temperatures must be finite and positive")
    if vi.shape == (3,) and len(species) == 1:
        vi = vi.reshape(1, 3)
    if vi.shape != (len(species), 3) or np.any(~np.isfinite(vi)) or np.any(np.linalg.norm(vi, axis=1) >= C):
        raise ValueError("ion_velocity_m_s must contain one finite subluminal vector per ion species")
    ui = vi @ qhat
    k = 4*np.pi*np.sin(angle/2)/(laser*1e-9)
    vphase = -C*(x-laser)/(2*laser*np.sin(angle/2))
    if (vgrid.ndim != 1 or fgrid.shape != vgrid.shape or vgrid.size < 9
            or np.any(~np.isfinite(vgrid)) or np.any(~np.isfinite(fgrid))
            or np.any(np.diff(vgrid) <= 0) or np.any(fgrid < 0)):
        raise ValueError("electron velocity grid and tabulated VDF must be matching finite non-negative arrays")
    # The susceptibility kernel includes the distributional endpoint jumps
    # for the declared zero extension outside [vgrid[0], vgrid[-1]].  Do not
    # reject non-zero endpoint values here: doing so would silently make the
    # forward model's accepted domain narrower than the corrected Landau
    # integral and would prevent physically explicit compact-support VDFs.
    evdf_area = float(_trapezoid(fgrid, vgrid))
    if not np.isfinite(evdf_area) or evdf_area <= 0:
        raise ValueError("tabulated electron VDF must have positive finite integral")
    fgrid = fgrid / evdf_area
    electron_pdf = np.interp(vphase-ue, vgrid, fgrid, left=0.0, right=0.0)
    chi_e = np.empty(x.size, dtype=complex)
    ion_chi, ion_pdf = [], []
    omega_pe2 = ne*E_CHARGE**2/(EPS0*ME)
    for j, vp in enumerate(vphase):
        chi_e[j] = generalized_electron_susceptibility(
            vgrid, fgrid, vp-ue, electron_density_m3=ne,
            wavenumber_m_inverse=k,
            pole_imaginary_part_m_s=pole_imaginary_part_m_s,
            landau_limit=landau_limit,
        )["electron_susceptibility"]
    for index, item in enumerate(species):
        density = fractions[index]*ne/item["charge_state"]
        sigma_i = np.sqrt(temperatures[index]*E_CHARGE/item["mass_kg"])
        omega_pi2 = density*(item["charge_state"]*E_CHARGE)**2/(EPS0*item["mass_kg"])
        pdf_i, _, _ = _maxwellian_pdf_and_derivatives(vphase-ui[index], sigma_i)
        chi_i = np.asarray([
            _landau_susceptibility(vphase[j]-ui[index], omega_pi2, k, sigma_i)
            for j in range(x.size)
        ], dtype=complex)
        ion_pdf.append(pdf_i)
        ion_chi.append(chi_i)
    ion_sum = np.sum(np.asarray(ion_chi), axis=0)
    dielectric = 1.0 + chi_e + ion_sum
    denominator = np.maximum(np.abs(dielectric)**2, np.finfo(float).tiny)
    electron_term = np.abs(1.0+ion_sum)**2/denominator*electron_pdf
    ion_term = np.zeros(x.size)
    for index, item in enumerate(species):
        ion_term += item["charge_state"]*fractions[index]*np.abs(chi_e)**2/denominator*ion_pdf[index]
    dynamic_structure = (2*np.pi/k)*(electron_term+ion_term)
    spectral = np.maximum(dynamic_structure, 0.0)*(2*np.pi*C*1e9/(x*x))
    if np.any(~np.isfinite(spectral)) or not np.any(spectral > 0):
        raise ValueError("tabulated-EVDF collective CTS model produced a non-finite or zero spectrum")
    return {
        "spectral_density_per_nm": spectral,
        "electron_susceptibility": chi_e,
        "ion_susceptibility": np.asarray(ion_chi),
        "dielectric_function": dielectric,
        "dynamic_structure_factor": dynamic_structure,
        "electron_flow_projection_m_s": ue,
        "ion_flow_projections_m_s": ui,
        "electron_distribution_model": "researcher_supplied_tabulated_projected_1D_EVDF",
        "electron_distribution_normalization": "normalized internally for susceptibility and spectral shape; absolute density is supplied separately",
        "model": "unmagnetized electrostatic collective Thomson scattering with generalized tabulated electron VDF and Maxwellian ions",
        "pole_quadrature": "piecewise_linear_first_order_Landau_pole_with_distributional_finite_support_edges",
        "velocity_support_extension": "zero_outside_supplied_interval; endpoint jumps included in susceptibility",
        "pole_imaginary_part_m_s": 0.0 if landau_limit else pole_imaginary_part_m_s,
        "landau_limit": bool(landau_limit),
        "limitations": [
            "projected 1-D electron VDF only; no unrestricted 3-D or anisotropic EVDF inversion",
            "non-relativistic unmagnetized electrostatic longitudinal response",
            "first-order Landau pole quadrature; no arbitrary-order pole closure",
            "finite velocity support is interpreted with an explicit zero extension; endpoint jumps are included; pole displacement is a numerical model input",
            "ion populations are Maxwellian and supplied externally",
        ],
        "experimental_validation": "not_performed",
    }
