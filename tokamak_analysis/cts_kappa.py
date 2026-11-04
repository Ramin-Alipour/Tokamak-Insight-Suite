"""Collisionless isotropic-Kappa electrostatic susceptibility."""
import numpy as np
from scipy.special import gammaln
from .thomson_generalized_dispersion import generalized_landau_integral
E=1.602176634e-19
EPS0=8.8541878128e-12
ME=9.1093837015e-31

def kappa_electron_susceptibility(v, phase_velocity_m_s, *, electron_density_m3,
        electron_temperature_eV, kappa, wavenumber_m_inverse,
        mass_kg=ME, pole_imaginary_part_m_s=None):
    """Exact 1-D marginal of isotropic 3-D Kappa; collisionless, unmagnetized."""
    v=np.asarray(v,dtype=float)
    ne,te,kap,k,m=map(float,(electron_density_m3,electron_temperature_eV,kappa,wavenumber_m_inverse,mass_kg))
    if v.ndim!=1 or v.size<9 or np.any(~np.isfinite(v)) or np.any(np.diff(v)<=0):
        raise ValueError("velocity grid must be finite, increasing, and contain at least 9 points")
    if not np.isfinite(ne) or ne<=0 or not np.isfinite(te) or te<=0 or not np.isfinite(k) or k<=0 or not np.isfinite(m) or m<=0:
        raise ValueError("density, temperature, wavenumber and mass must be positive and finite")
    if not np.isfinite(kap) or kap<=1.5:
        raise ValueError("kappa must be greater than 3/2")
    theta2=(2*kap-3)*E*te/(kap*m)
    a2=kap*theta2
    logn=gammaln(kap)-0.5*np.log(np.pi*a2)-gammaln(kap-0.5)
    pdf=np.exp(logn-kap*np.log1p(v*v/a2))
    out=generalized_landau_integral(v,pdf,phase_velocity_m_s,pole_imaginary_part_m_s=pole_imaginary_part_m_s)
    wp2=ne*E**2/(EPS0*m)
    out.update({"electron_susceptibility":complex(-wp2/k**2*out["integral"]),
        "plasma_frequency_squared_s_minus2":float(wp2),"wavenumber_m_inverse":k,
        "distribution_model":"exact_1D_marginal_of_isotropic_3D_Kappa","kappa":kap,
        "theta_m_s":float(np.sqrt(theta2)),"second_moment_m2_s2":float(E*te/m),
        "finite_velocity_grid_truncation":True,
        "limitations":["Collisionless unmagnetized longitudinal response only.",
        "No conserving collisions, FDT noise, magnetized response or Full-EM vertex.",
        "A 1-D marginal is not a reconstruction of the full 3-D EVDF."]})
    return out
