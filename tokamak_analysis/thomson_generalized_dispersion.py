"""Generalized unmagnetized electrostatic CTS response for tabulated 1-D EVDFs.

The susceptibility integral uses exact interval integration of the derivative
of a piecewise-linear projected VDF for the first-order Landau pole. This is a
transparent, Skolar-inspired real-axis quadrature primitive; it is not a
reimplementation of every arbitrary-order-pole feature in Skolar et al. (2025).
"""
from __future__ import annotations

import numpy as np

from .thomson_kinetic_models import (
    C, E_CHARGE, ME, EPS0, _resolve_ion_species,
    _maxwellian_pdf_and_derivatives, _landau_susceptibility,
)


def generalized_landau_integral(velocity_grid_m_s, probability_density_s_m,
                                phase_velocity_m_s, *, pole_imaginary_part_m_s=None):
    """Evaluate integral f'(v)/(v-zeta) dv for a tabulated projected VDF.

    f is linearly interpolated on the supplied velocity grid, so f' is
    piecewise constant and each interval integral is analytic. A positive
    imaginary pole displacement selects the retarded/Landau branch and avoids
    evaluating a singular real-axis logarithm exactly at a VDF knot.
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
    area = float(np.trapz(f, v))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("tabulated VDF must have positive finite integral")
    f = f / area
    eta = (1e-6 * float(np.median(np.diff(v))) if pole_imaginary_part_m_s is None
           else float(pole_imaginary_part_m_s))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("pole_imaginary_part_m_s must be positive and finite")
    zeta = complex(vp, eta)
    slopes = np.diff(f) / np.diff(v)
    integral = 0.0j
    for left, right, slope in zip(v[:-1], v[1:], slopes):
        integral += float(slope) * (np.log(complex(right, 0.0) - zeta)
                                    - np.log(complex(left, 0.0) - zeta))
    if not np.isfinite(integral.real) or not np.isfinite(integral.imag):
        raise ValueError("generalized Landau integral is non-finite")
    return {
        "integral": complex(integral),
        "normalized_probability_density_s_m": f,
        "pole": zeta,
        "pole_imaginary_part_m_s": eta,
        "quadrature": "analytic_interval_integral_of_piecewise_constant_derivative_of_piecewise_linear_VDF",
        "pole_order": 1,
        "limitation": "first-order Landau pole only; finite velocity support and positive pole displacement are explicit numerical inputs",
    }


def generalized_electron_susceptibility(
    velocity_grid_m_s, probability_density_s_m, phase_velocity_m_s, *,
    electron_density_m3, wavenumber_m_inverse, pole_imaginary_part_m_s=None,
):
    """Return longitudinal electron susceptibility for a tabulated 1-D VDF."""
    ne = float(electron_density_m3)
    k = float(wavenumber_m_inverse)
    if not np.isfinite(ne) or ne <= 0 or not np.isfinite(k) or k <= 0:
        raise ValueError("electron density and wavenumber must be positive and finite")
    response = generalized_landau_integral(
        velocity_grid_m_s, probability_density_s_m, phase_velocity_m_s,
        pole_imaginary_part_m_s=pole_imaginary_part_m_s,
    )
    omega_pe2 = ne * E_CHARGE**2 / (EPS0 * ME)
    chi = -(omega_pe2 / k**2) * response["integral"]
    response["electron_susceptibility"] = complex(chi)
    response["plasma_frequency_squared_s_minus2"] = float(omega_pe2)
    response["wavenumber_m_inverse"] = k
    return response


def collective_tabulated_electron_spectral_density(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
    electron_density_m3, electron_velocity_grid_m_s,
    electron_probability_density_s_m, ion_species, ion_fractions,
    ion_temperature_eV, electron_velocity_m_s, ion_velocity_m_s,
    incident_direction, scattered_direction, pole_imaginary_part_m_s=None,
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
    evdf_scale = max(float(np.max(fgrid)), np.finfo(float).tiny)
    if max(float(fgrid[0]), float(fgrid[-1])) > 1e-6 * evdf_scale:
        raise ValueError("tabulated electron VDF must decay to negligible values at both finite-support endpoints; otherwise omitted boundary delta contributions invalidate the susceptibility")
    evdf_area = float(np.trapz(fgrid, vgrid))
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
        "pole_quadrature": "piecewise_linear_first_order_Landau_pole",
        "pole_imaginary_part_m_s": pole_imaginary_part_m_s,
        "limitations": [
            "projected 1-D electron VDF only; no unrestricted 3-D or anisotropic EVDF inversion",
            "non-relativistic unmagnetized electrostatic longitudinal response",
            "first-order Landau pole quadrature; no arbitrary-order pole closure",
            "finite velocity support and pole displacement are numerical model inputs",
            "ion populations are Maxwellian and supplied externally",
        ],
        "experimental_validation": "not_performed",
    }
