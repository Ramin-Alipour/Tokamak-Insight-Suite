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