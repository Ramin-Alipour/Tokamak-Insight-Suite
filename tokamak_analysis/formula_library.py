"""Explicitly contracted implementations for foundational plasma formulae.

Inputs use SI unless the name states otherwise. Temperatures in eV are converted
using the exact elementary-charge value. These are analytic calculations, not
machine-specific empirical validations.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import trapezoid
from tis.constants import E_CHARGE, EPS0, M_E, MU0, C_LIGHT

def _finite(name, value):
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite.")
    return x

def _nonnegative(name, value):
    x=_finite(name,value)
    if x < 0: raise ValueError(f"{name} must be non-negative.")
    return x

def _positive(name, value):
    x=_finite(name,value)
    if x <= 0: raise ValueError(f"{name} must be positive.")
    return x

def electron_plasma_frequency(n_e_m3, *, sigma_n_e_m3=None):
    """Electron plasma frequency for local density n_e [m^-3]."""
    n=_positive("n_e_m3",n_e_m3)
    omega=math.sqrt(n*E_CHARGE**2/(EPS0*M_E))
    out={"omega_pe_rad_s":omega,"frequency_pe_Hz":omega/(2*math.pi)}
    if sigma_n_e_m3 is not None:
        s=_nonnegative("sigma_n_e_m3",sigma_n_e_m3)
        out["uncertainty_omega_pe_rad_s"]=0.5*omega*s/n
        out["uncertainty_frequency_pe_Hz"]=out["uncertainty_omega_pe_rad_s"]/(2*math.pi)
    return out

def debye_length(temperature_eV, density_m3, *, sigma_temperature_eV=None, sigma_density_m3=None):
    """Electron Debye length; temperature is in eV and density in m^-3."""
    T=_positive("temperature_eV",temperature_eV); n=_positive("density_m3",density_m3)
    if (sigma_temperature_eV is None) != (sigma_density_m3 is None):
        raise ValueError("Provide both temperature and density uncertainties, or neither.")
    lam=math.sqrt(EPS0*E_CHARGE*T/(n*E_CHARGE**2))
    out={"debye_length_m":lam}
    if sigma_temperature_eV is not None:
        sT=_nonnegative("sigma_temperature_eV",sigma_temperature_eV)
        sn=_nonnegative("sigma_density_m3",sigma_density_m3)
        out["uncertainty_debye_length_m"]=0.5*lam*math.sqrt((sT/T)**2+(sn/n)**2)
    return out

def electron_cyclotron_frequency(magnetic_field_T, *, sigma_magnetic_field_T=None):
    """Non-relativistic electron cyclotron frequency magnitude for B [T]."""
    B=abs(_finite("magnetic_field_T",magnetic_field_T))
    omega=E_CHARGE*B/M_E
    out={"omega_ce_rad_s":omega,"frequency_ce_Hz":omega/(2*math.pi)}
    if sigma_magnetic_field_T is not None:
        s=_nonnegative("sigma_magnetic_field_T",sigma_magnetic_field_T)
        out["uncertainty_omega_ce_rad_s"]=E_CHARGE*s/M_E
        out["uncertainty_frequency_ce_Hz"]=out["uncertainty_omega_ce_rad_s"]/(2*math.pi)
    return out

def ion_sound_speed(electron_temperature_eV, ion_temperature_eV, ion_mass_kg, gamma_i=1.0, *,
                    sigma_electron_temperature_eV=None, sigma_ion_temperature_eV=None,
                    sigma_ion_mass_kg=None, sigma_gamma_i=None):
    """Common closure c_s=sqrt(e*(Te + gamma_i*Ti)/mi); temperatures in eV."""
    Te=_positive("electron_temperature_eV",electron_temperature_eV)
    Ti=_nonnegative("ion_temperature_eV",ion_temperature_eV)
    mi=_positive("ion_mass_kg",ion_mass_kg); gamma=_nonnegative("gamma_i",gamma_i)
    sigmas=(sigma_electron_temperature_eV,sigma_ion_temperature_eV,sigma_ion_mass_kg,sigma_gamma_i)
    if any(v is not None for v in sigmas) and not all(v is not None for v in sigmas):
        raise ValueError("Provide all four independent input uncertainties, or none.")
    S=Te+gamma*Ti
    speed=math.sqrt(E_CHARGE*S/mi)
    out={"ion_sound_speed_m_s":speed}
    if sigma_electron_temperature_eV is not None:
        sTe,sTi,smi,sg=[_nonnegative(n,v) for n,v in zip(
            ("sigma_electron_temperature_eV","sigma_ion_temperature_eV","sigma_ion_mass_kg","sigma_gamma_i"),sigmas)]
        out["uncertainty_ion_sound_speed_m_s"]=math.sqrt(
            (speed*sTe/(2*S))**2+(speed*gamma*sTi/(2*S))**2+
            (speed*smi/(2*mi))**2+(speed*Ti*sg/(2*S))**2)
    return out

def electron_thermal_pressure(density_m3, temperature_eV, *, sigma_density_m3=None, sigma_temperature_eV=None):
    """Electron ideal-gas thermal pressure p_e=n_e*e*T_e[eV]."""
    n=_nonnegative("density_m3",density_m3); T=_nonnegative("temperature_eV",temperature_eV)
    if (sigma_density_m3 is None) != (sigma_temperature_eV is None):
        raise ValueError("Provide both density and temperature uncertainties, or neither.")
    p=n*E_CHARGE*T; out={"electron_pressure_Pa":p}
    if sigma_density_m3 is not None:
        sn=_nonnegative("sigma_density_m3",sigma_density_m3); sT=_nonnegative("sigma_temperature_eV",sigma_temperature_eV)
        out["uncertainty_electron_pressure_Pa"]=E_CHARGE*math.sqrt((T*sn)**2+(n*sT)**2)
    return out

def ion_thermal_pressure(species_density_m3, species_temperature_eV, *, sigma_species_density_m3=None, sigma_species_temperature_eV=None):
    """Sum ion pressure across explicitly supplied species arrays."""
    n=np.asarray(species_density_m3,dtype=float); T=np.asarray(species_temperature_eV,dtype=float)
    if n.ndim!=1 or T.shape!=n.shape or n.size==0 or not np.all(np.isfinite(n)) or not np.all(np.isfinite(T)) or np.any(n<0) or np.any(T<0):
        raise ValueError("Species density and temperature must be matching finite non-negative 1-D arrays.")
    if (sigma_species_density_m3 is None) != (sigma_species_temperature_eV is None):
        raise ValueError("Provide both species uncertainty arrays, or neither.")
    p=float(np.sum(n*E_CHARGE*T)); out={"ion_pressure_Pa":p}
    if sigma_species_density_m3 is not None:
        sn=np.asarray(sigma_species_density_m3,dtype=float); sT=np.asarray(sigma_species_temperature_eV,dtype=float)
        if sn.shape!=n.shape or sT.shape!=n.shape or not np.all(np.isfinite(sn)) or not np.all(np.isfinite(sT)) or np.any(sn<0) or np.any(sT<0):
            raise ValueError("Species uncertainty arrays must match species shape and be finite non-negative.")
        out["uncertainty_ion_pressure_Pa"]=float(E_CHARGE*np.sqrt(np.sum((T*sn)**2+(n*sT)**2)))
    return out

def total_thermal_pressure(electron_pressure_Pa, ion_pressure_Pa, *, sigma_electron_pressure_Pa=None, sigma_ion_pressure_Pa=None):
    """Sum electron and ion thermal pressures; inputs in Pa."""
    pe=_nonnegative("electron_pressure_Pa",electron_pressure_Pa); pi=_nonnegative("ion_pressure_Pa",ion_pressure_Pa)
    if (sigma_electron_pressure_Pa is None) != (sigma_ion_pressure_Pa is None):
        raise ValueError("Provide both pressure uncertainties, or neither.")
    out={"total_thermal_pressure_Pa":pe+pi}
    if sigma_electron_pressure_Pa is not None:
        se=_nonnegative("sigma_electron_pressure_Pa",sigma_electron_pressure_Pa); si=_nonnegative("sigma_ion_pressure_Pa",sigma_ion_pressure_Pa)
        out["uncertainty_total_thermal_pressure_Pa"]=math.hypot(se,si)
    return out

def _integrate_with_uncertainty(y, x, sigma_y, name):
    y=np.asarray(y,dtype=float); x=np.asarray(x,dtype=float)
    if y.ndim!=1 or x.shape!=y.shape or y.size<2 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0):
        raise ValueError(f"{name}: values and coordinate must be matching finite 1-D arrays; coordinate strictly increasing.")
    if np.any(y<0): raise ValueError(f"{name}: values must be non-negative.")
    value=float(trapezoid(y, x))
    out={"value":value}
    if sigma_y is not None:
        s=np.asarray(sigma_y,dtype=float)
        if s.shape!=y.shape or not np.all(np.isfinite(s)) or np.any(s<0):
            raise ValueError(f"{name}: uncertainty must match samples and be finite non-negative.")
        weights=np.empty_like(x); weights[0]=(x[1]-x[0])/2; weights[-1]=(x[-1]-x[-2])/2
        if len(x)>2: weights[1:-1]=(x[2:]-x[:-2])/2
        out["uncertainty"]=float(np.sqrt(np.sum((weights*s)**2)))
    return out

def thermal_stored_energy(pressure_Pa, volume_m3, *, sigma_pressure_Pa=None):
    """W_th=3/2 integral(p_th dV), with volume coordinate in m^3."""
    p=np.asarray(pressure_Pa,dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p<0): raise ValueError("Pressure must be finite and non-negative.")
    result=_integrate_with_uncertainty(p,volume_m3,sigma_pressure_Pa,"thermal_stored_energy")
    out={"thermal_stored_energy_J":1.5*result["value"]}
    if "uncertainty" in result: out["uncertainty_thermal_stored_energy_J"]=1.5*result["uncertainty"]
    return out

def line_integrated_density(density_m3, distance_m, *, sigma_density_m3=None):
    """Line integral of electron density along a path coordinate in metres."""
    result=_integrate_with_uncertainty(density_m3,distance_m,sigma_density_m3,"line_integrated_density")
    out={"line_integrated_density_m-2":result["value"]}
    if "uncertainty" in result: out["uncertainty_line_integrated_density_m-2"]=result["uncertainty"]
    return out


# The following helpers map the remaining catalogue formulae to explicit, unit-aware
# calculations. These functions are numerical primitives, not machine-specific
# empirical validations or substitutes for instrument calibration.

def plasma_frequency_from_density_cm3(density_cm3, *, sigma_density_cm3=None):
    """Electron plasma frequency from density in cm^-3; returns frequency in Hz."""
    n_cm3 = _positive("density_cm3", density_cm3)
    sigma_si = None if sigma_density_cm3 is None else _nonnegative("sigma_density_cm3", sigma_density_cm3) * 1.0e6
    result = electron_plasma_frequency(n_cm3 * 1.0e6, sigma_n_e_m3=sigma_si)
    out = {"frequency_pe_Hz": result["frequency_pe_Hz"]}
    if sigma_si is not None:
        out["uncertainty_frequency_pe_Hz"] = result["uncertainty_frequency_pe_Hz"]
    return out


def doppler_line_of_sight_velocity(delta_lambda, lambda0, *, sigma_delta_lambda=None, sigma_lambda0=None):
    """Non-relativistic line-of-sight velocity; wavelength shift and rest wavelength share units."""
    shift = _finite("delta_lambda", delta_lambda)
    rest = _positive("lambda0", lambda0)
    if (sigma_delta_lambda is None) != (sigma_lambda0 is None):
        raise ValueError("Provide both wavelength uncertainties, or neither.")
    velocity = C_LIGHT * shift / rest
    out = {"velocity_los_m_s": velocity}
    if sigma_delta_lambda is not None:
        sd = _nonnegative("sigma_delta_lambda", sigma_delta_lambda)
        sl = _nonnegative("sigma_lambda0", sigma_lambda0)
        out["uncertainty_velocity_los_m_s"] = math.hypot(C_LIGHT * sd / rest, C_LIGHT * shift * sl / rest**2)
    return out


def ion_temperature_from_doppler_sigma(sigma_lambda, lambda0, ion_mass_kg, *,
                                        sigma_sigma_lambda=None, sigma_lambda0=None, sigma_ion_mass_kg=None):
    """Thermal ion temperature from instrument-corrected Gaussian sigma; wavelengths share units."""
    sigma = _positive("sigma_lambda", sigma_lambda)
    rest = _positive("lambda0", lambda0)
    mass = _positive("ion_mass_kg", ion_mass_kg)
    uncertainties = (sigma_sigma_lambda, sigma_lambda0, sigma_ion_mass_kg)
    if any(v is not None for v in uncertainties) and not all(v is not None for v in uncertainties):
        raise ValueError("Provide all three independent uncertainties, or none.")
    ratio = sigma / rest
    temp = mass * C_LIGHT**2 / E_CHARGE * ratio**2
    out = {"ion_temperature_eV": temp}
    if sigma_sigma_lambda is not None:
        ss = _nonnegative("sigma_sigma_lambda", sigma_sigma_lambda)
        sl = _nonnegative("sigma_lambda0", sigma_lambda0)
        sm = _nonnegative("sigma_ion_mass_kg", sigma_ion_mass_kg)
        rel = math.sqrt((2 * ss / sigma)**2 + (2 * sl / rest)**2 + (sm / mass)**2)
        out["uncertainty_ion_temperature_eV"] = temp * rel
    return out


def gaussian_fwhm_from_sigma(sigma_width, *, sigma_sigma_width=None):
    """Convert Gaussian standard deviation to FWHM, preserving the width unit."""
    sigma = _nonnegative("sigma_width", sigma_width)
    factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    out = {"sigma_width": sigma, "fwhm_width": factor * sigma}
    if sigma_sigma_width is not None:
        s = _nonnegative("sigma_sigma_width", sigma_sigma_width)
        out["uncertainty_sigma_width"] = s
        out["uncertainty_fwhm_width"] = factor * s
    return out


def magnetic_energy_density(magnetic_field_T, *, sigma_magnetic_field_T=None):
    """Vacuum magnetic energy density B^2/(2 mu_0), in J m^-3."""
    B = _finite("magnetic_field_T", magnetic_field_T)
    value = B**2 / (2.0 * MU0)
    out = {"magnetic_energy_density_J_m3": value}
    if sigma_magnetic_field_T is not None:
        sB = _nonnegative("sigma_magnetic_field_T", sigma_magnetic_field_T)
        # First-order propagation through B^2/(2 mu_0) has zero derivative at B=0.
        # Returning a zero uncertainty there would be misleading when sigma_B > 0.
        if B == 0.0 and sB > 0.0:
            raise ValueError(
                "First-order magnetic-energy uncertainty is undefined at B=0 with non-zero "
                "field uncertainty; use a distribution-aware uncertainty model."
            )
        out["uncertainty_magnetic_energy_density_J_m3"] = abs(B) * sB / MU0
    return out


def toroidal_beta(pressure_Pa, toroidal_field_T, *, sigma_pressure_Pa=None, sigma_toroidal_field_T=None):
    """Local or explicitly averaged beta_t = p/(B_t^2/(2 mu_0)); pressure in Pa, field in T."""
    p = _nonnegative("pressure_Pa", pressure_Pa)
    B = _finite("toroidal_field_T", toroidal_field_T)
    if B == 0:
        raise ValueError("toroidal_field_T must be non-zero.")
    if (sigma_pressure_Pa is None) != (sigma_toroidal_field_T is None):
        raise ValueError("Provide both pressure and field uncertainties, or neither.")
    value = 2.0 * MU0 * p / B**2
    out = {"toroidal_beta": value}
    if sigma_pressure_Pa is not None:
        sp = _nonnegative("sigma_pressure_Pa", sigma_pressure_Pa)
        sB = _nonnegative("sigma_toroidal_field_T", sigma_toroidal_field_T)
        out["uncertainty_toroidal_beta"] = math.hypot(2.0 * MU0 * sp / B**2,
                                                        4.0 * MU0 * p * sB / abs(B)**3)
    return out


def safety_factor_large_aspect_ratio(r_m, major_radius_m, toroidal_field_T, poloidal_field_T, *,
                                     sigma_r_m=None, sigma_major_radius_m=None,
                                     sigma_toroidal_field_T=None, sigma_poloidal_field_T=None):
    """Large-aspect-ratio circular estimate q ~ r B_t/(R B_p), not an equilibrium reconstruction."""
    r = _nonnegative("r_m", r_m)
    R = _positive("major_radius_m", major_radius_m)
    Bt = _finite("toroidal_field_T", toroidal_field_T)
    Bp = _finite("poloidal_field_T", poloidal_field_T)
    if Bp == 0:
        raise ValueError("poloidal_field_T must be non-zero.")
    sigmas = (sigma_r_m, sigma_major_radius_m, sigma_toroidal_field_T, sigma_poloidal_field_T)
    if any(v is not None for v in sigmas) and not all(v is not None for v in sigmas):
        raise ValueError("Provide all four independent uncertainties, or none.")
    q = r * Bt / (R * Bp)
    out = {"safety_factor_estimate": q}
    if sigma_r_m is not None:
        sr, sR, sBt, sBp = [_nonnegative(n, v) for n, v in zip(
            ("sigma_r_m", "sigma_major_radius_m", "sigma_toroidal_field_T", "sigma_poloidal_field_T"), sigmas)]
        out["uncertainty_safety_factor_estimate"] = math.sqrt(
            (Bt * sr / (R * Bp))**2 + (q * sR / R)**2 +
            (r * sBt / (R * Bp))**2 + (q * sBp / Bp)**2)
    return out


def faraday_induction(flux_derivative_Wb_s, turns, *, sigma_flux_derivative_Wb_s=None):
    """Coil voltage from Faraday's law: V = -N dPhi/dt; derivative in Wb s^-1."""
    derivative = _finite("flux_derivative_Wb_s", flux_derivative_Wb_s)
    if isinstance(turns, (bool, np.bool_)):
        raise ValueError("turns must be a positive integer, not a boolean.")
    n_float = _positive("turns", turns)
    if not n_float.is_integer():
        raise ValueError("turns must be a positive integer.")
    out = {"coil_voltage_V": -int(n_float) * derivative}
    if sigma_flux_derivative_Wb_s is not None:
        s = _nonnegative("sigma_flux_derivative_Wb_s", sigma_flux_derivative_Wb_s)
        out["uncertainty_coil_voltage_V"] = int(n_float) * s
    return out


def plasma_loop_resistance_proxy(loop_voltage_V, plasma_current_A, *,
                                 sigma_loop_voltage_V=None, sigma_plasma_current_A=None):
    """Signed V_loop/I_p proxy in ohms; inductive and sign conventions remain the user's responsibility."""
    V = _finite("loop_voltage_V", loop_voltage_V)
    I = _finite("plasma_current_A", plasma_current_A)
    if I == 0:
        raise ValueError("plasma_current_A must be non-zero.")
    if (sigma_loop_voltage_V is None) != (sigma_plasma_current_A is None):
        raise ValueError("Provide both voltage and current uncertainties, or neither.")
    value = V / I
    out = {"loop_resistance_proxy_ohm": value}
    if sigma_loop_voltage_V is not None:
        sV = _nonnegative("sigma_loop_voltage_V", sigma_loop_voltage_V)
        sI = _nonnegative("sigma_plasma_current_A", sigma_plasma_current_A)
        out["uncertainty_loop_resistance_proxy_ohm"] = math.hypot(sV / I, V * sI / I**2)
    return out


def ohmic_power_approximation(loop_voltage_V, plasma_current_A, *,
                              sigma_loop_voltage_V=None, sigma_plasma_current_A=None):
    """Signed loop-voltage/current product in watts; not a general net-loss-power estimate."""
    V = _finite("loop_voltage_V", loop_voltage_V)
    I = _finite("plasma_current_A", plasma_current_A)
    if (sigma_loop_voltage_V is None) != (sigma_plasma_current_A is None):
        raise ValueError("Provide both voltage and current uncertainties, or neither.")
    value = V * I
    out = {"ohmic_power_approximation_W": value}
    if sigma_loop_voltage_V is not None:
        sV = _nonnegative("sigma_loop_voltage_V", sigma_loop_voltage_V)
        sI = _nonnegative("sigma_plasma_current_A", sigma_plasma_current_A)
        out["uncertainty_ohmic_power_approximation_W"] = math.hypot(I * sV, V * sI)
    return out


def energy_confinement_time(stored_energy_J, loss_power_W, *, sigma_stored_energy_J=None, sigma_loss_power_W=None):
    """Energy confinement-time estimate W_th/P_loss in seconds using a consistent loss-power definition."""
    W = _nonnegative("stored_energy_J", stored_energy_J)
    P = _positive("loss_power_W", loss_power_W)
    if (sigma_stored_energy_J is None) != (sigma_loss_power_W is None):
        raise ValueError("Provide both energy and power uncertainties, or neither.")
    value = W / P
    out = {"energy_confinement_time_s": value}
    if sigma_stored_energy_J is not None:
        sW = _nonnegative("sigma_stored_energy_J", sigma_stored_energy_J)
        sP = _nonnegative("sigma_loss_power_W", sigma_loss_power_W)
        out["uncertainty_energy_confinement_time_s"] = math.hypot(sW / P, W * sP / P**2)
    return out


def greenwald_density_reference(plasma_current_MA, minor_radius_m, *, sigma_plasma_current_MA=None, sigma_minor_radius_m=None):
    """Greenwald empirical reference: n_G = |I_p|[MA]/(pi a^2), in units of 10^20 m^-3."""
    raw_Ip = _finite("plasma_current_MA", plasma_current_MA)
    Ip = abs(raw_Ip)
    a = _positive("minor_radius_m", minor_radius_m)
    if (sigma_plasma_current_MA is None) != (sigma_minor_radius_m is None):
        raise ValueError("Provide both current and radius uncertainties, or neither.")
    if sigma_plasma_current_MA is not None and raw_Ip == 0.0 and _nonnegative(
        "sigma_plasma_current_MA", sigma_plasma_current_MA
    ) > 0.0:
        raise ValueError(
            "First-order Greenwald-density uncertainty is undefined at zero current with "
            "non-zero current uncertainty because the formula uses absolute current."
        )
    scaled = Ip / (math.pi * a**2)
    out = {"greenwald_density_1e20_m3": scaled, "greenwald_density_m3": scaled * 1.0e20}
    if sigma_plasma_current_MA is not None:
        sI = _nonnegative("sigma_plasma_current_MA", sigma_plasma_current_MA)
        sa = _nonnegative("sigma_minor_radius_m", sigma_minor_radius_m)
        s_scaled = math.hypot(sI / (math.pi * a**2), 2.0 * Ip * sa / (math.pi * a**3))
        out["uncertainty_greenwald_density_1e20_m3"] = s_scaled
        out["uncertainty_greenwald_density_m3"] = s_scaled * 1.0e20
    return out


def debye_parameter(density_m3, debye_length_m, *, sigma_density_m3=None, sigma_debye_length_m=None):
    """Number of particles in a Debye sphere, N_D=(4 pi/3)n_e lambda_D^3."""
    n = _positive("density_m3", density_m3)
    lam = _positive("debye_length_m", debye_length_m)
    if (sigma_density_m3 is None) != (sigma_debye_length_m is None):
        raise ValueError("Provide both density and Debye-length uncertainties, or neither.")
    value = (4.0 * math.pi / 3.0) * n * lam**3
    out = {"debye_parameter_particles": value}
    if sigma_density_m3 is not None:
        sn = _nonnegative("sigma_density_m3", sigma_density_m3)
        sl = _nonnegative("sigma_debye_length_m", sigma_debye_length_m)
        out["uncertainty_debye_parameter_particles"] = math.hypot((4.0 * math.pi / 3.0) * lam**3 * sn,
                                                                   4.0 * math.pi * n * lam**2 * sl)
    return out


def electron_plasma_angular_frequency(density_m3, *, sigma_density_m3=None):
    """Angular-frequency-only view of the electron plasma frequency for density in m^-3."""
    result = electron_plasma_frequency(density_m3, sigma_n_e_m3=sigma_density_m3)
    out = {"omega_pe_rad_s": result["omega_pe_rad_s"]}
    if "uncertainty_omega_pe_rad_s" in result:
        out["uncertainty_omega_pe_rad_s"] = result["uncertainty_omega_pe_rad_s"]
    return out


def ion_saturation_particle_flux_proxy(saturation_current_A, effective_area_m2, *,
                                       sigma_saturation_current_A=None, sigma_effective_area_m2=None):
    """Magnitude-based ion particle-flux proxy |I_sat|/(e A_eff), requiring a valid collection model."""
    raw_current = _finite("saturation_current_A", saturation_current_A)
    current = abs(raw_current)
    area = _positive("effective_area_m2", effective_area_m2)
    if (sigma_saturation_current_A is None) != (sigma_effective_area_m2 is None):
        raise ValueError("Provide both current and area uncertainties, or neither.")
    if sigma_saturation_current_A is not None and raw_current == 0.0 and _nonnegative(
        "sigma_saturation_current_A", sigma_saturation_current_A
    ) > 0.0:
        raise ValueError(
            "First-order uncertainty propagation through absolute current is undefined at zero "
            "current with non-zero uncertainty; use a distribution-aware uncertainty model."
        )
    value = current / (E_CHARGE * area)
    out = {"ion_saturation_particle_flux_m2_s": value}
    if sigma_saturation_current_A is not None:
        sI = _nonnegative("sigma_saturation_current_A", sigma_saturation_current_A)
        sA = _nonnegative("sigma_effective_area_m2", sigma_effective_area_m2)
        out["uncertainty_ion_saturation_particle_flux_m2_s"] = math.hypot(
            sI / (E_CHARGE * area), current * sA / (E_CHARGE * area**2))
    return out


def radiated_power_fraction(radiated_power_W, input_power_W, *, sigma_radiated_power_W=None, sigma_input_power_W=None):
    """Radiated-to-defined-input power ratio; values above unity are returned with a convention warning."""
    Pr = _nonnegative("radiated_power_W", radiated_power_W)
    Pin = _positive("input_power_W", input_power_W)
    if (sigma_radiated_power_W is None) != (sigma_input_power_W is None):
        raise ValueError("Provide both radiated- and input-power uncertainties, or neither.")
    value = Pr / Pin
    out = {"radiated_power_fraction": value}
    if value > 1.0:
        out["quality_flag"] = "fraction_above_unity_check_power_conventions_and_time_windows"
    if sigma_radiated_power_W is not None:
        sPr = _nonnegative("sigma_radiated_power_W", sigma_radiated_power_W)
        sPin = _nonnegative("sigma_input_power_W", sigma_input_power_W)
        out["uncertainty_radiated_power_fraction"] = math.hypot(sPr / Pin, Pr * sPin / Pin**2)
    return out


def integrated_surface_power(heat_flux_W_m2, area_weights_m2, *,
                             sigma_heat_flux_W_m2=None, sigma_area_weights_m2=None):
    """Integrate calibrated heat flux using matching surface-area weights; returns watts."""
    q = np.asarray(heat_flux_W_m2, dtype=float)
    area = np.asarray(area_weights_m2, dtype=float)
    if q.ndim != 1 or area.shape != q.shape or q.size == 0 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(area)) or np.any(area < 0):
        raise ValueError("Heat flux and area weights must be matching finite 1-D arrays; areas must be non-negative.")
    if (sigma_heat_flux_W_m2 is None) != (sigma_area_weights_m2 is None):
        raise ValueError("Provide both heat-flux and area uncertainties, or neither.")
    value = float(np.sum(q * area))
    out = {"integrated_surface_power_W": value}
    if sigma_heat_flux_W_m2 is not None:
        sq = np.asarray(sigma_heat_flux_W_m2, dtype=float)
        sa = np.asarray(sigma_area_weights_m2, dtype=float)
        if sq.shape != q.shape or sa.shape != q.shape or not np.all(np.isfinite(sq)) or not np.all(np.isfinite(sa)) or np.any(sq < 0) or np.any(sa < 0):
            raise ValueError("Uncertainty arrays must match the samples and be finite non-negative.")
        out["uncertainty_integrated_surface_power_W"] = float(np.sqrt(np.sum((area * sq)**2 + (q * sa)**2)))
    return out


def apparent_feature_speed(displacement_m, frame_interval_s, *, sigma_displacement_m=None, sigma_frame_interval_s=None):
    """Signed projected apparent feature speed, not necessarily plasma-flow velocity."""
    dx = _finite("displacement_m", displacement_m)
    dt = _positive("frame_interval_s", frame_interval_s)
    if (sigma_displacement_m is None) != (sigma_frame_interval_s is None):
        raise ValueError("Provide both displacement and interval uncertainties, or neither.")
    value = dx / dt
    out = {"apparent_feature_speed_m_s": value}
    if sigma_displacement_m is not None:
        sx = _nonnegative("sigma_displacement_m", sigma_displacement_m)
        st = _nonnegative("sigma_frame_interval_s", sigma_frame_interval_s)
        out["uncertainty_apparent_feature_speed_m_s"] = math.hypot(sx / dt, dx * st / dt**2)
    return out


def gas_pumping_speed(throughput_Pa_m3_s, pressure_Pa, *, sigma_throughput_Pa_m3_s=None, sigma_pressure_Pa=None):
    """Pumping speed Q/p in m^3 s^-1 for throughput in Pa m^3 s^-1 and pressure in Pa."""
    Q = _nonnegative("throughput_Pa_m3_s", throughput_Pa_m3_s)
    p = _positive("pressure_Pa", pressure_Pa)
    if (sigma_throughput_Pa_m3_s is None) != (sigma_pressure_Pa is None):
        raise ValueError("Provide both throughput and pressure uncertainties, or neither.")
    value = Q / p
    out = {"gas_pumping_speed_m3_s": value}
    if sigma_throughput_Pa_m3_s is not None:
        sQ = _nonnegative("sigma_throughput_Pa_m3_s", sigma_throughput_Pa_m3_s)
        sp = _nonnegative("sigma_pressure_Pa", sigma_pressure_Pa)
        out["uncertainty_gas_pumping_speed_m3_s"] = math.hypot(sQ / p, Q * sp / p**2)
    return out


def injected_particle_number(particle_rate_s, time_s, *, sigma_particle_rate_s=None):
    """Integrate calibrated particle injection rate [s^-1] over strictly increasing time [s]."""
    result = _integrate_with_uncertainty(particle_rate_s, time_s, sigma_particle_rate_s, "injected_particle_number")
    out = {"injected_particle_number": result["value"]}
    if "uncertainty" in result:
        out["uncertainty_injected_particle_number"] = result["uncertainty"]
    return out


def particle_confinement_time_proxy(stored_particles, loss_rate_s, *, sigma_stored_particles=None, sigma_loss_rate_s=None):
    """Particle confinement-time proxy N_p/dN_loss/dt in seconds."""
    N = _nonnegative("stored_particles", stored_particles)
    rate = _positive("loss_rate_s", loss_rate_s)
    if (sigma_stored_particles is None) != (sigma_loss_rate_s is None):
        raise ValueError("Provide both stored-particle and loss-rate uncertainties, or neither.")
    value = N / rate
    out = {"particle_confinement_time_s": value}
    if sigma_stored_particles is not None:
        sN = _nonnegative("sigma_stored_particles", sigma_stored_particles)
        sr = _nonnegative("sigma_loss_rate_s", sigma_loss_rate_s)
        out["uncertainty_particle_confinement_time_s"] = math.hypot(sN / rate, N * sr / rate**2)
    return out


def relative_fluctuation_level(signal):
    """Population RMS fluctuation divided by the absolute mean; DC mean must be safely non-zero."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be a finite 1-D array with at least two samples.")
    mean = float(np.mean(x))
    scale = float(np.max(np.abs(x)))
    if scale == 0.0 or abs(mean) <= np.finfo(float).eps * scale * 10.0:
        raise ValueError("signal mean is zero or too close to zero for a stable relative fluctuation.")
    fluctuation_rms = float(np.sqrt(np.mean((x - mean)**2)))
    return {"relative_fluctuation_level": fluctuation_rms / abs(mean),
            "mean_signal": mean, "fluctuation_rms": fluctuation_rms}


def rms_signal(signal, *, sigma_signal=None):
    """Root-mean-square of finite samples, including any DC component."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be a non-empty finite 1-D array.")
    value = float(np.sqrt(np.mean(x**2)))
    out = {"rms_signal": value}
    if sigma_signal is not None:
        s = np.asarray(sigma_signal, dtype=float)
        if s.shape != x.shape or not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ValueError("sigma_signal must match samples and be finite non-negative.")
        if value == 0.0:
            raise ValueError("RMS uncertainty linearization is undefined when the measured RMS is zero.")
        out["uncertainty_rms_signal"] = float(np.sqrt(np.sum((x * s / (x.size * value))**2)))
    return out


def nyquist_frequency(sampling_frequency_Hz, *, sigma_sampling_frequency_Hz=None):
    """Nyquist frequency for uniformly sampled data; anti-alias filtering is a separate requirement."""
    fs = _positive("sampling_frequency_Hz", sampling_frequency_Hz)
    out = {"nyquist_frequency_Hz": fs / 2.0}
    if sigma_sampling_frequency_Hz is not None:
        sfs = _nonnegative("sigma_sampling_frequency_Hz", sigma_sampling_frequency_Hz)
        out["uncertainty_nyquist_frequency_Hz"] = sfs / 2.0
    return out


def cross_correlation_lag(signal_x, signal_y, sample_interval_s):
    """Return lag of the maximum positive cross-correlation; positive means y lags x."""
    x = np.asarray(signal_x, dtype=float)
    y = np.asarray(signal_y, dtype=float)
    dt = _positive("sample_interval_s", sample_interval_s)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("signals must be matching finite 1-D arrays with at least three samples.")
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    norm = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if norm == 0.0:
        raise ValueError("cross-correlation is undefined for a constant signal.")
    corr = np.correlate(y0, x0, mode="full") / norm
    lags = np.arange(-x.size + 1, x.size)
    index = int(np.argmax(corr))
    return {"cross_correlation_lag_s": float(lags[index] * dt),
            "normalized_peak_correlation": float(corr[index])}


def magnitude_squared_coherence(signal_x, signal_y, sampling_frequency_Hz, *, nperseg=None):
    """Welch magnitude-squared coherence; estimator uncertainty is not inferred from one result."""
    from scipy.signal import coherence
    x = np.asarray(signal_x, dtype=float)
    y = np.asarray(signal_y, dtype=float)
    fs = _positive("sampling_frequency_Hz", sampling_frequency_Hz)
    if x.ndim != 1 or y.shape != x.shape or x.size < 8 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("signals must be matching finite 1-D arrays with at least eight samples.")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise ValueError("Coherence is undefined when either input signal is constant.")
    if nperseg is None:
        segment = min(256, x.size)
    else:
        raw_segment = _finite("nperseg", nperseg)
        if not raw_segment.is_integer():
            raise ValueError("nperseg must be an integer number of samples.")
        segment = int(raw_segment)
    if segment < 2 or segment > x.size:
        raise ValueError("nperseg must be between 2 and the signal length.")
    frequencies, values = coherence(x, y, fs=fs, nperseg=segment)
    if not np.all(np.isfinite(values)):
        raise ValueError("Coherence estimator returned non-finite values; check signal power and numerical conditioning.")
    # The estimator is mathematically bounded by [0, 1]; clip only floating-point roundoff.
    values = np.clip(values, 0.0, 1.0)
    return {"coherence_frequency_Hz": frequencies.tolist(),
            "magnitude_squared_coherence": values.tolist(),
            "nperseg": segment,
            "uncertainty_note": "No confidence interval estimated; depends on segment count, windowing and stationarity."}


def bolometry_regularized_inverse(measurements, geometry_matrix, regularization, *, sigma_measurements=None):
    """Unconstrained Tikhonov estimate x=(A.T A + lambda^2 I)^-1 A.T y; negative x is retained and flagged."""
    y = np.asarray(measurements, dtype=float)
    A = np.asarray(geometry_matrix, dtype=float)
    lam = _positive("regularization", regularization)
    if y.ndim != 1 or A.ndim != 2 or A.shape[0] != y.size or A.shape[1] < 1 or y.size < 1:
        raise ValueError("measurements must be 1-D and match the row count of a 2-D geometry matrix.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(A)):
        raise ValueError("measurements and geometry matrix must be finite.")
    # SVD form avoids explicitly solving normal equations, which square the
    # condition number. This is the same unconstrained Tikhonov estimator.
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    factors = singular_values / (singular_values**2 + lam**2)
    kernel = (Vh.T * factors) @ U.T
    x = kernel @ y
    regularized_eigenvalues = singular_values**2 + lam**2
    if A.shape[1] > singular_values.size:
        regularized_eigenvalues = np.concatenate(
            [regularized_eigenvalues, np.full(A.shape[1] - singular_values.size, lam**2)]
        )
    out = {"emissivity_estimate": x.tolist(),
           "residual_norm": float(np.linalg.norm(A @ x - y)),
           "regularization": lam,
           "normal_matrix_condition_number": float(np.max(regularized_eigenvalues) / np.min(regularized_eigenvalues)),
           "negative_emissivity_voxel_count": int(np.count_nonzero(x < 0)),
           "reconstruction_note": "Unconstrained linear Tikhonov estimate; geometry/calibration and regularization bias are not validated by this calculation."}
    if sigma_measurements is not None:
        sy = np.asarray(sigma_measurements, dtype=float)
        if sy.ndim == 0:
            sy = np.full(y.shape, float(sy))
        if sy.shape != y.shape or not np.all(np.isfinite(sy)) or np.any(sy < 0):
            raise ValueError("sigma_measurements must be a non-negative scalar or match the measurement vector.")
        covariance = (kernel * (sy**2)[None, :]) @ kernel.T
        out["uncertainty_emissivity_estimate"] = np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist()
        out["uncertainty_note"] = "Conditional linear propagation of measurement uncertainty only; excludes geometry and regularization-model uncertainty."
    return out


def particle_balance_rate(input_rate_s, output_rate_s, source_rate_s, loss_rate_s, *,
                          sigma_input_rate_s=None, sigma_output_rate_s=None,
                          sigma_source_rate_s=None, sigma_loss_rate_s=None):
    """Net particle inventory rate dN/dt = input - output + source - loss; all rates in s^-1."""
    names = ("input_rate_s", "output_rate_s", "source_rate_s", "loss_rate_s")
    vals = [_nonnegative(n, v) for n, v in zip(names, (input_rate_s, output_rate_s, source_rate_s, loss_rate_s))]
    sigmas = (sigma_input_rate_s, sigma_output_rate_s, sigma_source_rate_s, sigma_loss_rate_s)
    if any(v is not None for v in sigmas) and not all(v is not None for v in sigmas):
        raise ValueError("Provide all four independent rate uncertainties, or none.")
    value = vals[0] - vals[1] + vals[2] - vals[3]
    out = {"particle_balance_rate_s": value}
    if sigma_input_rate_s is not None:
        s = [_nonnegative(n, v) for n, v in zip(
            ("sigma_input_rate_s", "sigma_output_rate_s", "sigma_source_rate_s", "sigma_loss_rate_s"), sigmas)]
        out["uncertainty_particle_balance_rate_s"] = math.sqrt(sum(v**2 for v in s))
    return out

def langmuir_electron_temperature_from_slope(slope_per_V, *, sigma_slope_per_V=None):
    """Maxwellian electron-retarding branch: Te[eV] = 1 / slope[V^-1]."""
    slope = _positive("slope_per_V", slope_per_V)
    temperature = 1.0 / slope
    out = {"electron_temperature_eV": temperature}
    if sigma_slope_per_V is not None:
        sigma_slope = _nonnegative("sigma_slope_per_V", sigma_slope_per_V)
        out["uncertainty_electron_temperature_eV"] = sigma_slope / slope**2
    return out


def langmuir_ion_density_from_saturation_current(
    ion_saturation_current_A, electron_temperature_eV, probe_area_m2,
    ion_mass_kg, sheath_coefficient, *, charge_state=1,
    sigma_ion_saturation_current_A=None, sigma_electron_temperature_eV=None,
    sigma_probe_area_m2=None, sigma_ion_mass_kg=None,
    sigma_sheath_coefficient=None,
):
    """Single-ion Bohm/sheath-model density inversion from calibrated ion-saturation current."""
    current = _positive("ion_saturation_current_magnitude_A", abs(_finite("ion_saturation_current_A", ion_saturation_current_A)))
    temperature = _positive("electron_temperature_eV", electron_temperature_eV)
    area = _positive("probe_area_m2", probe_area_m2)
    mass = _positive("ion_mass_kg", ion_mass_kg)
    coefficient = _positive("sheath_coefficient", sheath_coefficient)
    if isinstance(charge_state, bool) or not isinstance(charge_state, int) or charge_state < 1:
        raise ValueError("charge_state must be a positive integer.")
    density = current / (
        coefficient * E_CHARGE * area * charge_state**1.5
        * math.sqrt(E_CHARGE * temperature / mass)
    )
    out = {"ion_density_m3": density}
    sigmas = (
        sigma_ion_saturation_current_A, sigma_electron_temperature_eV,
        sigma_probe_area_m2, sigma_ion_mass_kg, sigma_sheath_coefficient,
    )
    if any(value is not None for value in sigmas) and not all(value is not None for value in sigmas):
        raise ValueError("Provide all five independent input uncertainties, or none.")
    if sigma_ion_saturation_current_A is not None:
        si, sT, sA, sm, sC = [
            _nonnegative(name, value) for name, value in zip(
                ("sigma_ion_saturation_current_A", "sigma_electron_temperature_eV",
                 "sigma_probe_area_m2", "sigma_ion_mass_kg",
                 "sigma_sheath_coefficient"), sigmas
            )
        ]
        relative = math.sqrt(
            (si / current)**2 + (sT / (2.0 * temperature))**2
            + (sA / area)**2 + (sm / (2.0 * mass))**2
            + (sC / coefficient)**2
        )
        out["uncertainty_ion_density_m3"] = density * relative
    return out

