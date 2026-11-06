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
    ion_temperature_eV=None, electron_velocity_m_s=None, ion_velocity_m_s=None,
    incident_direction, scattered_direction, pole_imaginary_part_m_s=None,
    landau_limit=False, ion_velocity_grids_m_s=None,
    ion_probability_densities_s_m=None,
):
    """Unmagnetized electrostatic CTS spectrum with kinetic projected EVDFs.

    Electrons use a supplied 1-D marginal along q. Ions may use the legacy
    Maxwellian closure or independent supplied 1-D projected distributions in
    each species rest frame. Tabulated ions use the same Landau/Plemelj
    susceptibility integral and finite-support endpoint-jump convention as
    electrons. This remains a 1-D electrostatic forward model.
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
    tabulated_ions = ion_velocity_grids_m_s is not None or ion_probability_densities_s_m is not None
    temperatures = None if ion_temperature_eV is None else np.asarray(np.atleast_1d(ion_temperature_eV), dtype=float)
    vi = np.asarray(ion_velocity_m_s, dtype=float)
    if fractions.shape != (len(species),):
        raise ValueError("ion_fractions must provide one value per species")
    if (np.any(~np.isfinite(fractions)) or np.any(fractions <= 0)
            or not np.isclose(np.sum(fractions), 1.0, rtol=1e-6, atol=1e-8)):
        raise ValueError("ion fractions must be finite, positive and sum to one")
    if not tabulated_ions:
        if temperatures is None or temperatures.shape != (len(species),):
            raise ValueError("ion_temperature_eV must provide one value per species for the Maxwellian ion closure")
        if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0):
            raise ValueError("Maxwellian ion temperatures must be finite and positive")
    elif temperatures is not None and (
        temperatures.shape != (len(species),) or np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0)
    ):
        raise ValueError("optional ion_temperature_eV, when supplied with tabulated ions, must be finite, positive and species-sized")
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
    if np.any(np.abs(vgrid) >= C):
        raise ValueError(
            "the non-relativistic tabulated-Landau CTS kernel requires the entire "
            "projected electron velocity support to satisfy |v_parallel| < c; "
            "use a relativistic kinetic response for superluminal-support grids"
        )
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
    if tabulated_ions and (ion_velocity_grids_m_s is None or ion_probability_densities_s_m is None):
        raise ValueError("ion_velocity_grids_m_s and ion_probability_densities_s_m must be supplied together")
    if tabulated_ions:
        ion_grids = np.asarray(ion_velocity_grids_m_s, dtype=float)
        ion_distributions = np.asarray(ion_probability_densities_s_m, dtype=float)
        if (ion_grids.ndim != 2 or ion_distributions.shape != ion_grids.shape
                or ion_grids.shape[0] != len(species) or ion_grids.shape[1] < 9
                or np.any(~np.isfinite(ion_grids)) or np.any(~np.isfinite(ion_distributions))
                or np.any(ion_distributions < 0) or np.any(np.diff(ion_grids, axis=1) <= 0)):
            raise ValueError("tabulated ion grids and distributions must be matching finite arrays shaped (n_ion_species,n_velocity), with increasing grids and non-negative densities")
        if np.any(np.abs(ion_grids) >= C):
            raise ValueError(
                "the non-relativistic tabulated-Landau CTS kernel requires every "
                "projected ion velocity support to satisfy |v_parallel| < c"
            )
        ion_areas = np.asarray([_trapezoid(row, grid) for row, grid in zip(ion_distributions, ion_grids)])
        if np.any(~np.isfinite(ion_areas)) or np.any(ion_areas <= 0):
            raise ValueError("each tabulated ion distribution must have positive finite integral")
        ion_distributions = ion_distributions / ion_areas[:, None]
    else:
        ion_grids = ion_distributions = None
    for index, item in enumerate(species):
        density = fractions[index]*ne/item["charge_state"]
        omega_pi2 = density*(item["charge_state"]*E_CHARGE)**2/(EPS0*item["mass_kg"])
        if tabulated_ions:
            grid_i = ion_grids[index]
            pdf_i = np.interp(vphase-ui[index], grid_i, ion_distributions[index], left=0.0, right=0.0)
            chi_i = np.asarray([
                -(omega_pi2/k**2)*generalized_landau_integral(
                    grid_i, ion_distributions[index], vphase[j]-ui[index],
                    pole_imaginary_part_m_s=pole_imaginary_part_m_s,
                    landau_limit=landau_limit,
                )["integral"]
                for j in range(x.size)
            ], dtype=complex)
        else:
            sigma_i = np.sqrt(temperatures[index]*E_CHARGE/item["mass_kg"])
            pdf_i, _, _ = _maxwellian_pdf_and_derivatives(vphase-ui[index], sigma_i)
            chi_i = np.asarray([
                _landau_susceptibility(vphase[j]-ui[index], omega_pi2, k, sigma_i)
                for j in range(x.size)
            ], dtype=complex)
        ion_pdf.append(pdf_i)
        ion_chi.append(chi_i)

    # Derive directional central moments from the supplied projected EVDFs.
    # These are q-parallel moments, not unrestricted 3-D temperatures.
    def _projected_moments(grid, density, mass_kg):
        area = float(_trapezoid(density, grid))
        normalized = np.asarray(density, dtype=float) / area
        mean = float(_trapezoid(grid * normalized, grid))
        centered = grid - mean
        variance = float(_trapezoid(centered**2 * normalized, grid))
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("projected EVDF must have a positive finite second central moment")
        third = float(_trapezoid(centered**3 * normalized, grid))
        fourth = float(_trapezoid(centered**4 * normalized, grid))
        return {
            "mean_velocity_relative_to_species_frame_m_s": mean,
            "variance_m2_s2": variance,
            "directional_second_moment_temperature_eV": float(mass_kg * variance / E_CHARGE),
            "standardized_skewness": float(third / variance**1.5),
            "excess_kurtosis": float(fourth / variance**2 - 3.0),
            "moment_definition": "central_moments_of_normalized_projected_1D_EVDF",
        }

    electron_moments = _projected_moments(vgrid, fgrid, ME)
    ion_moments = []
    for index, item in enumerate(species):
        if tabulated_ions:
            moments = _projected_moments(ion_grids[index], ion_distributions[index], item["mass_kg"])
            moments["distribution_source"] = "researcher_supplied_tabulated_projected_1D_EVDF"
        else:
            variance = temperatures[index] * E_CHARGE / item["mass_kg"]
            moments = {
                "mean_velocity_relative_to_species_frame_m_s": 0.0,
                "variance_m2_s2": float(variance),
                "directional_second_moment_temperature_eV": float(temperatures[index]),
                "standardized_skewness": 0.0,
                "excess_kurtosis": 0.0,
                "moment_definition": "analytic_maxwellian_central_moments",
                "distribution_source": "Maxwellian_per_species",
            }
        moments["species"] = item["name"]
        moments["bulk_flow_projection_m_s"] = float(ui[index])
        moments["total_mean_projection_m_s"] = float(ui[index] + moments["mean_velocity_relative_to_species_frame_m_s"])
        ion_moments.append(moments)
    electron_moments["bulk_flow_projection_m_s"] = float(ue)
    electron_moments["total_mean_projection_m_s"] = float(ue + electron_moments["mean_velocity_relative_to_species_frame_m_s"])

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
        "electron_projected_moments": electron_moments,
        "ion_projected_moments": ion_moments,
        "ion_projected_pdf_at_resonance": np.asarray(ion_pdf),
        "electron_distribution_model": "researcher_supplied_tabulated_projected_1D_EVDF",
        "ion_distribution_model": (
            "researcher_supplied_tabulated_projected_1D_EVDF_per_species"
            if tabulated_ions else "Maxwellian_per_species"
        ),
        "ion_distribution_normalization": (
            "each species rest-frame projected EVDF normalized internally; ion density remains set by quasineutral composition"
            if tabulated_ions else "Maxwellian normalized analytically"
        ),
        "electron_distribution_normalization": "normalized internally for susceptibility and spectral shape; absolute density is supplied separately",
        "model": (
            "unmagnetized electrostatic collective Thomson scattering with generalized tabulated electron and species-resolved tabulated ion VDFs"
            if tabulated_ions else
            "unmagnetized electrostatic collective Thomson scattering with generalized tabulated electron VDF and Maxwellian ions"
        ),
        "pole_quadrature": "piecewise_linear_first_order_Landau_pole_with_distributional_finite_support_edges",
        "velocity_support_extension": "zero_outside_supplied_interval; endpoint jumps included in susceptibility",
        "pole_imaginary_part_m_s": 0.0 if landau_limit else pole_imaginary_part_m_s,
        "landau_limit": bool(landau_limit),
        "limitations": [
            "projected 1-D electron VDF only; no unrestricted 3-D or anisotropic EVDF inversion",
            "non-relativistic unmagnetized electrostatic longitudinal response",
            "first-order Landau pole quadrature; no arbitrary-order pole closure",
            "finite velocity support is interpreted with an explicit zero extension; endpoint jumps are included; pole displacement is a numerical model input",
            "tabulated ion populations, when supplied, are independent 1-D rest-frame marginals; no unrestricted 3-D ion EVDF is inferred",
        ],
        "experimental_validation": "not_performed",
    }



def collective_multi_population_tabulated_electron_spectral_density(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
    electron_density_m3, electron_populations, **common_plasma_geometry,
):
    """Electrostatic CTS spectrum for an arbitrary finite mixture of projected electron EVDFs.

    Each population contributes its own density-weighted Landau susceptibility
    and its own Klimontovich fluctuation source.  The total dielectric is
    epsilon=1+sum_j(chi_ej)+sum_i(chi_ii), while the electron source is
    sum_j(alpha_j f_j(v_phase-u_j)); replacing this by one averaged EVDF is
    generally incorrect because susceptibility and fluctuation source enter
    the dynamic structure factor differently.

    Population mappings require velocity_grid_m_s, probability_density_s_m,
    and density_fraction. bulk_velocity_m_s is an optional 3-vector. The
    distributions are rest-frame 1-D marginals along the scattering vector.
    This is an unmagnetized, non-relativistic, electrostatic forward model;
    it does not infer the component fractions or unrestricted 3-D EVDFs.
    """
    if not isinstance(electron_populations, (list, tuple)) or len(electron_populations) < 2:
        raise ValueError("electron_populations must contain at least two explicit electron populations")
    ne = float(electron_density_m3)
    if not np.isfinite(ne) or ne <= 0.0:
        raise ValueError("electron_density_m3 must be positive and finite")
    x = np.asarray(wavelength_nm, dtype=float)
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    if x.ndim != 1 or x.size < 9 or np.any(~np.isfinite(x)) or np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("wavelength_nm must be a positive, finite, strictly increasing vector with at least 9 bins")
    if not np.isfinite(laser) or laser <= 0.0 or not np.isfinite(angle) or not 0.0 < angle <= np.pi:
        raise ValueError("laser wavelength and scattering angle are invalid")
    incident = np.asarray(common_plasma_geometry.get("incident_direction"), dtype=float)
    scattered = np.asarray(common_plasma_geometry.get("scattered_direction"), dtype=float)
    if incident.shape != (3,) or scattered.shape != (3,) or np.any(~np.isfinite(incident)) or np.any(~np.isfinite(scattered)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(incident) == 0.0 or np.linalg.norm(scattered) == 0.0:
        raise ValueError("incident and scattered directions must be non-zero")
    incident /= np.linalg.norm(incident)
    scattered /= np.linalg.norm(scattered)
    measured_angle = float(np.arccos(np.clip(np.dot(incident, scattered), -1.0, 1.0)))
    if not np.isclose(measured_angle, angle, rtol=0.0, atol=1e-5):
        raise ValueError("scattering_angle_rad is inconsistent with incident/scattered directions")
    qhat = scattered - incident
    qhat /= np.linalg.norm(qhat)
    c_light = 299792458.0
    fractions = []
    normalized_populations = []
    for index, population in enumerate(electron_populations):
        if not isinstance(population, Mapping):
            raise ValueError(f"electron_populations[{index}] must be a mapping")
        fraction = float(population.get("density_fraction", np.nan))
        grid = np.asarray(population.get("velocity_grid_m_s"), dtype=float)
        pdf = np.asarray(population.get("probability_density_s_m"), dtype=float)
        bulk = np.asarray(population.get("bulk_velocity_m_s", (0.0, 0.0, 0.0)), dtype=float)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError(f"electron_populations[{index}].density_fraction must be positive and finite")
        if grid.ndim != 1 or grid.size < 9 or pdf.shape != grid.shape:
            raise ValueError(f"electron_populations[{index}] requires matching 1-D velocity and probability-density arrays with >=9 samples")
        if np.any(~np.isfinite(grid)) or np.any(~np.isfinite(pdf)) or np.any(np.diff(grid) <= 0.0) or np.any(pdf < 0.0):
            raise ValueError(f"electron_populations[{index}] velocity grid must increase and its PDF must be finite and non-negative")
        if bulk.shape != (3,) or np.any(~np.isfinite(bulk)) or np.linalg.norm(bulk) >= c_light:
            raise ValueError(f"electron_populations[{index}].bulk_velocity_m_s must be a finite subluminal 3-vector")
        drift = float(np.dot(bulk, qhat))
        if np.any(np.abs(grid + drift) >= c_light):
            raise ValueError("the non-relativistic multi-population Landau kernel requires all lab-frame projected supports to satisfy |v_parallel| < c")
        area = float(_trapezoid(pdf, grid))
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError(f"electron_populations[{index}] PDF must have positive finite integral")
        fractions.append(fraction)
        normalized_populations.append({
            "velocity_grid_m_s": grid,
            "probability_density_s_m": pdf / area,
            "density_fraction": fraction,
            "bulk_velocity_m_s": bulk,
            "bulk_flow_projection_m_s": drift,
        })
    fractions = np.asarray(fractions, dtype=float)
    if not np.isclose(np.sum(fractions), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("electron population density_fraction values must sum to one")
    # Use one common retarded-pole displacement for every electron component
    # and the ion response. Letting each differently sampled EVDF choose its
    # own default eta would introduce a spurious population-dependent damping.
    common = dict(common_plasma_geometry)
    if not bool(common.get("landau_limit", False)) and common.get("pole_imaginary_part_m_s") is None:
        common["pole_imaginary_part_m_s"] = 1.0e-6 * min(
            float(np.median(np.diff(item["velocity_grid_m_s"])))
            for item in normalized_populations
        )
    # The single-population call supplies the common ion response, ion
    # fluctuation PDFs, wavelength mapping and validated composition contract.
    seed = normalized_populations[0]
    if "electron_velocity_m_s" in common:
        raise ValueError("electron bulk flows belong in each electron_populations entry, not electron_velocity_m_s")
    result = collective_tabulated_electron_spectral_density(
        x, laser, angle,
        electron_density_m3=ne,
        electron_velocity_grid_m_s=seed["velocity_grid_m_s"],
        electron_probability_density_s_m=seed["probability_density_s_m"],
        electron_velocity_m_s=(0.0, 0.0, 0.0),
        **common,
    )
    k = 4.0*np.pi*np.sin(angle/2.0)/(laser*1e-9)
    phase_velocity = -c_light*(x-laser)/(2.0*laser*np.sin(angle/2.0))
    chi_e_populations = []
    electron_source = np.zeros(x.size, dtype=float)
    population_moments = []
    for index, population in enumerate(normalized_populations):
        drift = population["bulk_flow_projection_m_s"]
        susceptibility = np.asarray([
            generalized_electron_susceptibility(
                population["velocity_grid_m_s"],
                population["probability_density_s_m"],
                phase_velocity_value - drift,
                electron_density_m3=ne*population["density_fraction"],
                wavenumber_m_inverse=k,
                pole_imaginary_part_m_s=common.get("pole_imaginary_part_m_s"),
                landau_limit=bool(common.get("landau_limit", False)),
            )["electron_susceptibility"]
            for phase_velocity_value in phase_velocity
        ], dtype=complex)
        chi_e_populations.append(susceptibility)
        electron_source += population["density_fraction"] * np.interp(
            phase_velocity - drift,
            population["velocity_grid_m_s"],
            population["probability_density_s_m"],
            left=0.0, right=0.0,
        )
        grid = population["velocity_grid_m_s"]
        pdf = population["probability_density_s_m"]
        mean = float(_trapezoid(grid*pdf, grid))
        variance = float(_trapezoid((grid-mean)**2*pdf, grid))
        if variance <= 0.0 or not np.isfinite(variance):
            raise ValueError(f"electron population {index} must have a positive finite second central moment")
        population_moments.append({
            "population_index": index,
            "density_fraction": float(population["density_fraction"]),
            "bulk_flow_projection_m_s": float(drift),
            "mean_velocity_relative_to_population_frame_m_s": mean,
            "total_mean_projection_m_s": float(drift+mean),
            "directional_second_moment_temperature_eV": float(9.1093837015e-31*variance/1.602176634e-19),
            "standardized_skewness": float(_trapezoid((grid-mean)**3*pdf, grid)/variance**1.5),
            "excess_kurtosis": float(_trapezoid((grid-mean)**4*pdf, grid)/variance**2-3.0),
        })
    # Combine moments in the laboratory frame with the law of total
    # cumulants; simply averaging component temperatures would omit the
    # between-population drift/variance contribution.
    mixture_mean = float(sum(
        item["density_fraction"] * (
            item["bulk_flow_projection_m_s"] + moments["mean_velocity_relative_to_population_frame_m_s"]
        )
        for item, moments in zip(normalized_populations, population_moments)
    ))
    mixture_variance = 0.0
    mixture_third = 0.0
    mixture_fourth = 0.0
    mean_bulk_flow = 0.0
    for item, moments in zip(normalized_populations, population_moments):
        alpha = item["density_fraction"]
        component_mean = item["bulk_flow_projection_m_s"] + moments["mean_velocity_relative_to_population_frame_m_s"]
        delta = component_mean - mixture_mean
        variance = moments["directional_second_moment_temperature_eV"] * 1.602176634e-19 / 9.1093837015e-31
        skew = moments["standardized_skewness"]
        excess = moments["excess_kurtosis"]
        third_central = skew * variance**1.5
        fourth_central = (excess + 3.0) * variance**2
        mixture_variance += alpha * (variance + delta**2)
        mixture_third += alpha * (third_central + 3.0*delta*variance + delta**3)
        mixture_fourth += alpha * (fourth_central + 4.0*delta*third_central
                                   + 6.0*delta**2*variance + delta**4)
        mean_bulk_flow += alpha * item["bulk_flow_projection_m_s"]
    if mixture_variance <= 0.0 or not np.isfinite(mixture_variance):
        raise ValueError("multi-population electron mixture has a non-positive or non-finite variance")
    mixture_electron_moments = {
        "mean_velocity_relative_to_species_frame_m_s": mixture_mean - mean_bulk_flow,
        "variance_m2_s2": mixture_variance,
        "directional_second_moment_temperature_eV": float(9.1093837015e-31*mixture_variance/1.602176634e-19),
        "standardized_skewness": float(mixture_third/mixture_variance**1.5),
        "excess_kurtosis": float(mixture_fourth/mixture_variance**2-3.0),
        "bulk_flow_projection_m_s": float(mean_bulk_flow),
        "total_mean_projection_m_s": float(mixture_mean),
        "moment_definition": "density-weighted laboratory-frame moments including between-population drift variance",
    }
    chi_e_populations = np.asarray(chi_e_populations)
    chi_e_total = np.sum(chi_e_populations, axis=0)
    chi_i = np.asarray(result["ion_susceptibility"])
    chi_i_total = np.sum(chi_i, axis=0)
    dielectric = 1.0 + chi_e_total + chi_i_total
    denominator = np.maximum(np.abs(dielectric)**2, np.finfo(float).tiny)
    electron_term = np.abs(1.0+chi_i_total)**2/denominator*electron_source
    ion_term = np.zeros(x.size, dtype=float)
    ion_pdf = np.asarray(result["ion_projected_pdf_at_resonance"])
    ion_fractions = np.asarray(common["ion_fractions"], dtype=float)
    ion_species = _resolve_ion_species(common["ion_species"])
    for index, item in enumerate(ion_species):
        ion_term += item["charge_state"]*ion_fractions[index]*np.abs(chi_e_total)**2/denominator*ion_pdf[index]
    dynamic_structure = (2.0*np.pi/k)*(electron_term+ion_term)
    spectral = np.maximum(dynamic_structure, 0.0)*(2.0*np.pi*c_light*1e9/(x*x))
    if np.any(~np.isfinite(spectral)) or not np.any(spectral > 0.0):
        raise ValueError("multi-population CTS model produced a non-finite or zero spectrum")
    result.update({
        "spectral_density_per_nm": spectral,
        "electron_susceptibility": chi_e_total,
        "electron_population_susceptibility": chi_e_populations,
        "dielectric_function": dielectric,
        "dynamic_structure_factor": dynamic_structure,
        "electron_population_moments": population_moments,
        "electron_projected_moments": mixture_electron_moments,
        "electron_flow_projection_m_s": float(mean_bulk_flow),
        "electron_population_density_fractions": fractions.copy(),
        "electron_projected_pdf_at_resonance": electron_source,
        "electron_distribution_model": "multi_population_tabulated_projected_1D_EVDF",
        "model": "unmagnetized electrostatic CTS with density-weighted multi-population projected electron EVDFs and species-resolved ion response",
        "limitations": list(result["limitations"]) + [
            "electron population fractions and component EVDFs are supplied inputs, not inferred by this forward model",
            "only 1-D projected non-relativistic electron marginals are supported; no unrestricted 3-D or relativistic population inversion",
        ],
    })
    return result
