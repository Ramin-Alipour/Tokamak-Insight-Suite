"""Explicitly contracted implementations for foundational plasma formulae.

Inputs use SI unless the name states otherwise. Temperatures in eV are converted
using the exact elementary-charge value. These are analytic calculations, not
machine-specific empirical validations.
"""
from __future__ import annotations
import math
import numpy as np
from tis.constants import E_CHARGE, EPS0, M_E, MU0

def _finite(name, value):
    x=float(value)
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
        if s/n > float("inf"): raise ValueError("Invalid density uncertainty.")
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

def ion_sound_speed(electron_temperature_eV, ion_temperature_eV, ion_mass_kg, gamma_i=1.0):
    """Common closure c_s=sqrt(e*(Te + gamma_i*Ti)/mi), temperatures in eV."""
    Te=_positive("electron_temperature_eV",electron_temperature_eV)
    Ti=_nonnegative("ion_temperature_eV",ion_temperature_eV)
    mi=_positive("ion_mass_kg",ion_mass_kg); gamma=_nonnegative("gamma_i",gamma_i)
    return {"ion_sound_speed_m_s":math.sqrt(E_CHARGE*(Te+gamma*Ti)/mi)}

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
    value=float(np.trapezoid(y,x))
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
