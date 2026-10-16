"""Ion temperature from instrument-corrected Doppler FWHM."""
import numpy as np
from scipy.constants import atomic_mass
from scipy.optimize import brentq
C=299792458.0; EV=1.602176634e-19; U=1.66053906660e-27
def ion_temperature_from_doppler_fwhm(fwhm_nm, rest_wavelength_nm, ion_mass_amu, *, instrumental_fwhm_nm=0.0, mass_number=None):
    vals=[fwhm_nm,rest_wavelength_nm,ion_mass_amu,instrumental_fwhm_nm]
    if not all(np.isfinite(v) for v in vals) or fwhm_nm<=0 or rest_wavelength_nm<=0 or ion_mass_amu<=0 or instrumental_fwhm_nm<0: raise ValueError("Widths/wavelength/mass must be valid and positive")
    if fwhm_nm<=instrumental_fwhm_nm: raise ValueError("Observed width must exceed instrumental width")
    # Gaussian broadening subtraction in quadrature.
    thermal_width_nm=np.sqrt(fwhm_nm**2-instrumental_fwhm_nm**2)
    sigma_v=C*(thermal_width_nm/rest_wavelength_nm)/2.354820045
    mass=ion_mass_amu*U
    Ti_eV=mass*sigma_v**2/EV
    return {"ion_temperature_eV":float(Ti_eV),"thermal_fwhm_nm":float(thermal_width_nm),"velocity_sigma_m_s":float(sigma_v),"ion_mass_amu":float(ion_mass_amu),"caveat":"Assumes a Gaussian line, thermal Doppler broadening only after quadrature instrument correction, and negligible unresolved flow/Zeeman/Stark broadening."}

def thermal_doppler_fwhm_nm(ion_temperature_eV, rest_wavelength_nm, ion_mass_amu):
    """Forward thermal Doppler relation: FWHM_lambda=lambda0/c*sqrt(8 ln2*kT/m)."""
    ti,lam,mass=np.broadcast_arrays(np.asarray(ion_temperature_eV,dtype=float),np.asarray(rest_wavelength_nm,dtype=float),np.asarray(ion_mass_amu,dtype=float))
    if any(not np.all(np.isfinite(x)) for x in (ti,lam,mass)) or np.any(ti<0) or np.any(lam<=0) or np.any(mass<=0):
        raise ValueError("Temperature must be non-negative; wavelength and ion mass positive; all finite")
    width=lam*1e-9/C*np.sqrt(8*np.log(2)*ti*EV/(mass*U))*1e9
    return {"thermal_fwhm_nm":width,"equation":"Delta_lambda_FWHM=lambda0/c*sqrt(8*ln(2)*e*Ti_eV/(mi*u))",
            "assumption":"Pure thermal Gaussian Doppler broadening; instrumental and non-thermal widths are excluded."}
